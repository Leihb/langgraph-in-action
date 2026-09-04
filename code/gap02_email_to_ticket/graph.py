"""空白 2：非结构化邮件 → 结构化工单。

    read_email → extract → validate ─┬─ FORMAT 且还有次数 → extract（带着错误重抽）
                    ▲                 ├─ MISSING → ask_customer → END（等客人回信，同一线程下一封邮件接着跑）
                    │                 ├─ MISMATCH → human_review（interrupt）→ file / discard
                    └─────────────────┴─ 都没问题 → file（归并或新建工单，回执）→ END

一个线程（conversation_id）一个 thread_id。客人回信是同一线程的新一封邮件，图从 START 再跑一遍，
state 里已经有前面的邮件，extract 看的是整条线程。

跟例子 1 的分工：例子 1 读邮件是为了**回复**（分类 → 查资料 → 拟稿）；这一篇读邮件是为了
**落库**（抽字段 → 校验 → 补缺 → 归并）。真实系统两者都要，前者面向客人，后者面向坐席。
"""

import json
import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from common.llm import chat_model
from gap02_email_to_ticket import tools
from gap02_email_to_ticket.orders import get_order
from gap02_email_to_ticket.prompts import EXTRACT, FEEDBACK
from gap02_email_to_ticket.schema import SLA_HOURS, Problem, TicketDraft, is_merchant, priority_for, validate

json_llm = chat_model().with_structured_output(None, method="json_mode")
MAX_EXTRACT_ATTEMPTS = 3


class TicketState(TypedDict):
    conversation_id: str
    emails: Annotated[list[dict], operator.add]   # 这条线程收到的全部邮件
    incoming: dict                                 # 这一次触发的那封
    draft: TicketDraft | None
    problems: list[Problem]
    attempts: int
    ticket_no: str | None
    status: str | None                             # filed / waiting_customer / discarded / ignored
    trail: Annotated[list[str], operator.add]
    model_calls: Annotated[int, operator.add]


def read_email(state: TicketState) -> Command[Literal["extract", "file"]]:
    m = state["incoming"]
    line = f"── 收到 {m['id']}（{m['conversation_id']}）{m['from']}：{m['subject']}"
    if state.get("ticket_no"):
        # 这条线程已经有工单了，后续邮件直接追加到工单上，不再抽字段
        return Command(goto="file", update={"emails": [m], "attempts": 0, "trail": [line]})
    return Command(goto="extract", update={"emails": [m], "attempts": 0, "problems": [], "trail": [line]})


def extract(state: TicketState) -> dict:
    thread = "\n\n".join(f"[{e['received_at']}] 发件人 {e['from']}\n主题：{e['subject']}\n{e['body']}" for e in state["emails"])
    feedback = ""
    if state.get("problems"):
        feedback = FEEDBACK.format(problems="\n".join(f"- {p['field']}：{p['message']}" for p in state["problems"]))
    raw = json_llm.invoke(EXTRACT.format(thread=thread, feedback=feedback))
    draft = {k: raw.get(k) for k in TicketDraft.__annotations__}
    n = state["attempts"] + 1
    shown = {k: v for k, v in draft.items() if v not in (None, "")}
    return {"draft": draft, "attempts": n, "model_calls": 1,
            "trail": [f"extract（第 {n} 次）→ {shown}"]}


def check(state: TicketState) -> Command[Literal["extract", "ask_customer", "human_review", "file", "__end__"]]:
    """校验是代码。三类问题三条路，先看最严重的。"""
    d, m = state["draft"], state["incoming"]
    if d.get("ticket_type") == "not_a_request":
        return Command(goto=END, update={"status": "ignored", "problems": [], "trail": ["check：不是诉求，忽略"]})
    problems = validate(d, sender=m["from"], received_at=m["received_at"])
    kinds = {p["kind"] for p in problems}
    if not problems:
        return Command(goto="file", update={"problems": [], "trail": ["check：全部通过"]})
    summary = "；".join(f"{p['kind']} {p['field']}" for p in problems)
    if "FORMAT" in kinds:
        if state["attempts"] < MAX_EXTRACT_ATTEMPTS:
            return Command(goto="extract", update={"problems": problems, "trail": [f"check：{summary} → 喂回去重抽"]})
        return Command(goto="human_review", update={"problems": problems, "trail": [f"check：{summary}，重抽 {MAX_EXTRACT_ATTEMPTS} 次仍不行 → 转人工"]})
    if "MISMATCH" in kinds:
        return Command(goto="human_review", update={"problems": problems, "trail": [f"check：{summary} → 转人工"]})
    return Command(goto="ask_customer", update={"problems": problems, "trail": [f"check：{summary} → 问客人"]})


def ask_customer(state: TicketState) -> dict:
    m = state["incoming"]
    missing = [p for p in state["problems"] if p["kind"] == "MISSING"]
    asks = {"order_id": "您的订单号（形如 KL-778，在确认邮件里能找到）",
            "target_date": "您希望改到哪一天", "reason": "退改的原因"}
    body = ("您好，我们收到了您的邮件。为了尽快处理，请补充以下信息：\n"
            + "\n".join(f"- {asks.get(p['field'], p['field'])}" for p in missing)
            + "\n\n直接回复本邮件即可。")
    tools.send_email(m["from"], f"Re: {m['subject']}", body)
    return {"status": "waiting_customer", "trail": [f"ask_customer：已回信问 {[p['field'] for p in missing]}，等回复"]}


def human_review(state: TicketState) -> Command[Literal["file", "__end__"]]:
    decision = interrupt({"conversation_id": state["conversation_id"], "from": state["incoming"]["from"],
                          "draft": {k: v for k, v in state["draft"].items() if v not in (None, "")},
                          "problems": [p["message"] for p in state["problems"]],
                          "options": ["file", "discard"]})
    if str(decision).strip().lower().startswith("file"):
        return Command(goto="file", update={"trail": ["human_review：坐席决定照样建单"]})
    return Command(goto=END, update={"status": "discarded", "trail": [f"human_review：坐席决定丢弃——{decision}"]})


def file(state: TicketState) -> dict:
    """归并或新建。优先级、SLA、发件人邮箱都是代码填的。"""
    d, m = state["draft"], state["incoming"]
    if state.get("ticket_no"):
        tools.append_update(state["ticket_no"], state["conversation_id"], f"{m['from']}：{m['body']}")
        return {"status": "filed", "trail": [f"file：线程已有工单 {state['ticket_no']}，追加为更新"]}
    existing = tools.find_open_ticket(d.get("order_id"), d.get("category"))
    if existing:
        tools.append_update(existing["ticket_no"], state["conversation_id"], f"{m['from']}：{m['body']}")
        tools.send_email(m["from"], f"Re: {m['subject']}", f"您好，您的来信已并入工单 {existing['ticket_no']}，我们会一起处理。")
        return {"ticket_no": existing["ticket_no"], "status": "filed",
                "trail": [f"file：订单 {d['order_id']} 已有同类工单 {existing['ticket_no']}（另一条线程），归并进去"]}
    priority = priority_for(d, m["received_at"])
    ticket = {**d, "ticket_type": "merchant_request" if is_merchant(m["from"]) else d["ticket_type"],
              "customer_email": m["from"], "conversation_id": state["conversation_id"],
              "priority": priority, "sla_hours": SLA_HOURS[priority], "received_at": m["received_at"],
              "product": (get_order(d.get("order_id") or "") or {}).get("product")}
    no = tools.create_ticket(ticket)
    if ticket["ticket_type"] != "not_a_request":
        tools.send_email(m["from"], f"Re: {m['subject']}", f"您好，已为您建立工单 {no}，我们会在 {SLA_HOURS[priority]} 小时内跟进。")
    return {"ticket_no": no, "status": "filed",
            "trail": [f"file：新建 {no}  {ticket['ticket_type']}/{ticket['category']}  订单 {d.get('order_id')}  优先级 {priority}（SLA {SLA_HOURS[priority]}h）"]}


def build_graph(checkpointer):
    g = StateGraph(TicketState)
    for name, fn in [("read_email", read_email), ("extract", extract), ("check", check),
                     ("ask_customer", ask_customer), ("human_review", human_review), ("file", file)]:
        g.add_node(name, fn)
    g.add_edge(START, "read_email")
    g.add_edge("extract", "check")
    g.add_edge("ask_customer", END)
    g.add_edge("file", END)
    return g.compile(checkpointer=checkpointer)
