"""例子 1：客服邮件分流。官方 "Thinking in LangGraph" 那个邮件 agent 的重做。

读邮件 → 分类 → 按类别走三条路之一（查文档 / 建 bug 单 / 直接转人工）→ 拟稿
→ 紧急的先给人看，不紧急的直接发。

跟前面 15 期最大的写法差别：**路由写在节点里**。节点返回 `Command(goto=...)`
同时带上状态更新，一个函数既说"我改了什么"也说"下一步去哪"，图上就不用
再单独挂 add_conditional_edges。哪种写法好不是原则问题：分叉多、每个分叉
的判断只跟这个节点自己算出来的结果有关时，写在节点里读起来更顺。
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from common.llm import chat_model
from ex01_email_triage import tools
from ex01_email_triage.prompts import CLASSIFY, DRAFT
from ex01_email_triage.state import EmailClassification, EmailState

llm = chat_model()
# 分类要的是结构化结果：让模型按 EmailClassification 这个结构吐字段，省掉
# "吐一段话再用正则去捞"。with_structured_output 底下有三种协议，这台端点
# （DeepSeek）真机试下来两种走不通：默认的 json_schema 回 400 "This
# response_format type is unavailable now"；function_calling 把结构当成一个
# 工具让模型强制调用，思考模式的模型回 400 "Thinking mode does not support
# this tool_choice"。剩下 json_mode：只要求模型返回合法 JSON，字段是什么
# 靠提示词说清楚、靠下面 _validate 兜底。换端点要重新试一遍这三种。
classifier = llm.with_structured_output(EmailClassification, method="json_mode")

INTENTS = ("question", "bug", "billing", "feature", "complex")
URGENCIES = ("low", "medium", "high", "critical")


def _validate(raw: dict) -> EmailClassification:
    """json_mode 不保证字段取值合法，拿不准的一律往"要人看"的方向归。"""
    intent = raw.get("intent") if raw.get("intent") in INTENTS else "complex"
    urgency = raw.get("urgency") if raw.get("urgency") in URGENCIES else "high"
    return {"intent": intent, "urgency": urgency,
            "topic": str(raw.get("topic", "")), "summary": str(raw.get("summary", ""))}

import json
from pathlib import Path

EMAILS = {e["id"]: e for e in json.loads((Path(__file__).parent / "data" / "emails.json").read_text())}


def read_email(state: EmailState) -> dict:
    """真实场景这里是拉邮箱 API；这里从假数据里取。只写原始字段，不做任何加工。"""
    email = EMAILS[state["email_id"]]
    return {"sender": email["from"], "subject": email["subject"], "body": email["body"],
            "trace": [f"read_email <- {state['email_id']}"]}


def classify_intent(state: EmailState) -> Command[Literal["human_review", "search_documentation", "bug_tracking", "draft_response"]]:
    result = _validate(classifier.invoke(
        CLASSIFY.format(sender=state["sender"], subject=state["subject"], body=state["body"])
    ))
    intent, urgency = result["intent"], result["urgency"]
    # 路由规则是代码，不是模型：模型只给出 intent/urgency 两个标签，
    # "什么标签走哪条路"写死在下面四行里，改规则不用改提示词。
    if intent == "billing" or urgency == "critical" or intent == "complex":
        goto = "human_review"
    elif intent in ("question", "feature"):
        goto = "search_documentation"
    elif intent == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"
    return Command(update={"classification": result, "trace": [f"classify -> {intent}/{urgency} -> {goto}"]}, goto=goto)


def search_documentation(state: EmailState) -> Command[Literal["draft_response"]]:
    c = state["classification"]
    results = tools.search_docs(f"{c['topic']} {c['summary']}")
    return Command(update={"search_results": results, "trace": [f"search_docs -> {len(results)} 条"]}, goto="draft_response")


def bug_tracking(state: EmailState) -> Command[Literal["draft_response"]]:
    c = state["classification"]
    ticket_id = tools.create_ticket(c["summary"], state["sender"], c["urgency"])
    return Command(update={"ticket_id": ticket_id, "trace": [f"create_ticket -> {ticket_id}"]}, goto="draft_response")


def draft_response(state: EmailState) -> Command[Literal["human_review", "send_reply"]]:
    c = state["classification"]
    # 上下文在这里才拼：state 里存的是列表和 id，拼成什么样是这个节点自己的事
    context_parts = []
    if state.get("search_results"):
        context_parts.append("可参考的文档：\n" + "\n".join(f"- {r}" for r in state["search_results"]))
    if state.get("ticket_id"):
        context_parts.append(f"已为这个问题建了工单 {state['ticket_id']}，请在回复里告知客人并说明会跟进。")
    if not context_parts:
        context_parts.append("没有额外资料，按常识礼貌回复，不要承诺具体结果。")
    reply = llm.invoke(DRAFT.format(subject=state["subject"], body=state["body"], intent=c["intent"],
                                    urgency=c["urgency"], context="\n\n".join(context_parts)))
    goto = "human_review" if c["urgency"] in ("high", "critical") else "send_reply"
    return Command(update={"draft": reply.content.strip(), "trace": [f"draft -> {goto}"]}, goto=goto)


def human_review(state: EmailState) -> Command[Literal["send_reply", "draft_response", "__end__"]]:
    """interrupt() 必须是这个节点的第一个动作：恢复时整个节点从头重跑，
    放在它前面的代码会执行两遍。

    官方例子里这个节点批准后直接去 send_reply。但 billing/complex 两条路是在
    拟稿之前就转人工的，这时 state 里没有草稿——照抄会发出一封正文为 None 的
    邮件。所以这里多一条路：没有草稿的批准，先去 draft_response 拟稿，拟完
    按紧急程度它还会回到这里再审一次。"""
    decision = interrupt({
        "email_id": state["email_id"],
        "from": state["sender"],
        "subject": state["subject"],
        "classification": state["classification"],
        "draft": state.get("draft"),
        "options": ["approve", "edit:<新草稿>", "reject"],
    })
    if decision == "approve":
        if not state.get("draft"):
            return Command(update={"review": "approve", "trace": ["human_review -> approve（无草稿，先拟稿）"]}, goto="draft_response")
        return Command(update={"review": "approve", "trace": ["human_review -> approve"]}, goto="send_reply")
    if isinstance(decision, str) and decision.startswith("edit:"):
        return Command(update={"review": "edit", "draft": decision[5:].strip(), "trace": ["human_review -> edit"]}, goto="send_reply")
    return Command(update={"review": "reject", "trace": ["human_review -> reject"]}, goto=END)


def send_reply(state: EmailState) -> dict:
    tools.send_email(state["sender"], f"Re: {state['subject']}", state["draft"])
    return {"sent": True, "trace": ["send_reply -> 已发"]}


def build_graph(checkpointer):
    builder = StateGraph(EmailState)
    builder.add_node("read_email", read_email)
    builder.add_node("classify_intent", classify_intent)
    # 文档库会偶发超时——这类"再试一次就好"的错误交给 RetryPolicy，不写进节点逻辑
    builder.add_node("search_documentation", search_documentation,
                     retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.2, retry_on=tools.TransientError))
    builder.add_node("bug_tracking", bug_tracking)
    builder.add_node("draft_response", draft_response)
    builder.add_node("human_review", human_review)
    builder.add_node("send_reply", send_reply)
    builder.add_edge(START, "read_email")
    builder.add_edge("read_email", "classify_intent")
    builder.add_edge("send_reply", END)
    # 其余的边都由节点里的 Command(goto=...) 决定，这里不用再画
    return builder.compile(checkpointer=checkpointer)
