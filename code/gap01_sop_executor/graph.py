"""SOP 执行器：一张图跑任何一份 SOP 文件。

    plan ──qa──▶ qa ──▶ END
      └──sop──▶ step ──▶ step ──▶ ... ──▶ finish ──▶ END
                 ▲ (approve 被拒 → 回退到指定步骤)

三个节点用模型（plan / qa / finish），一个节点不用（step）。step 每次执行 SOP 里的一步，
用 Command(goto="step") 把自己接回来；ask 和 approve 两种步骤会在第一行 interrupt()。
"""

import json
import operator
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from common.llm import chat_model
from gap01_sop_executor.prompts import EXTRACT, PLAN, QA, SUMMARY
from gap01_sop_executor.rules import RULES
from gap01_sop_executor.sop import SOP_DIR, load_sops, resolve_args, step_index, when_holds
from gap01_sop_executor.tools import TOOLS

SOPS = load_sops()
POLICY = (SOP_DIR / "policy.md").read_text(encoding="utf-8")
llm = chat_model()
json_llm = chat_model().with_structured_output(None, method="json_mode")  # 例子 1 的结论：这台端点只有 json_mode 通


class SopState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    sop: str | None
    cursor: int                                  # 下一步的下标
    facts: dict                                  # 一路收集的事实：订单、金额、余额、审批人……
    trail: Annotated[list[str], operator.add]    # 执行记录，一步一行
    outcome: str | None                          # done / stopped / None
    model_calls: Annotated[int, operator.add]


# ---------- 模型出场一：听懂坐席要干什么 ----------

def plan(state: SopState) -> Command[Literal["step", "qa"]]:
    text = state["messages"][-1].content
    sop_list = "\n".join(f"- {s['name']}：{s['description']}（字段：{', '.join(s['fields'])}）" for s in SOPS.values())
    raw = json_llm.invoke(PLAN.format(sops=sop_list, text=text))
    facts = {k: v for k, v in (raw.get("facts") or {}).items() if v not in (None, "", "null")}
    if raw.get("mode") == "sop" and raw.get("sop") in SOPS:
        return Command(goto="step", update={
            "mode": "sop", "sop": raw["sop"], "cursor": 0, "facts": facts, "outcome": None, "model_calls": 1,
            "trail": [f"── 新任务：{text}", f"plan：走 SOP「{SOPS[raw['sop']]['title']}」，已知 {facts}"],
        })
    return Command(goto="qa", update={"mode": "qa", "sop": None, "model_calls": 1, "trail": [f"── 提问：{text}"]})


# ---------- 模型出场二：开放问答 ----------

def qa(state: SopState) -> dict:
    answer = llm.invoke(QA.format(policy=POLICY, text=state["messages"][-1].content)).content
    return {"messages": [AIMessage(answer)], "outcome": "done", "model_calls": 1}


# ---------- 不用模型：执行 SOP 的一步 ----------

def step(state: SopState) -> Command[Literal["step", "finish"]]:
    sop = SOPS[state["sop"]]
    i = state["cursor"]
    if i >= len(sop["steps"]):
        return Command(goto="finish", update={"outcome": "done"})
    s = sop["steps"][i]
    facts = dict(state["facts"])
    nxt = {"cursor": i + 1}

    if not when_holds(s, facts):
        return Command(goto="step", update={**nxt, "trail": [f"{s['id']}：条件 `{s['when']}` 不成立，跳过"]})

    if s["kind"] == "call":
        try:
            result = TOOLS[s["tool"]](**resolve_args(s, facts))
        except Exception as e:  # noqa: BLE001
            return Command(goto="finish", update={"outcome": "stopped",
                                                  "trail": [f"{s['id']}：调 {s['tool']} 失败——{e}"]})
        facts[s["save_as"]] = result
        shown = result if not isinstance(result, dict) else {k: v for k, v in result.items() if k not in ("compensations",)}
        return Command(goto="step", update={**nxt, "facts": facts, "trail": [f"{s['id']}：{s['tool']} → {shown}"]})

    if s["kind"] == "check":
        for name in s["rules"]:
            ok, why = RULES[name](facts)
            if not ok:
                return Command(goto="finish", update={"outcome": "stopped", "trail": [f"{s['id']}：规则 {name} 不过——{why}"]})
        return Command(goto="step", update={**nxt, "trail": [f"{s['id']}：{len(s['rules'])} 条规则全过"]})

    if s["kind"] == "ask":
        missing = [f for f in s["fields"] if facts.get(f) in (None, "")]
        if not missing:
            return Command(goto="step", update={**nxt, "trail": [f"{s['id']}：字段齐了，不用问"]})
        # 第一行 interrupt：节点恢复时从头重跑，上面只有纯读取，没有副作用
        answer = interrupt({"kind": "ask", "step": s["id"], "missing": missing, "prompt": s["prompt"]})
        raw = json_llm.invoke(EXTRACT.format(prompt=s["prompt"], text=answer, fields=", ".join(missing)))
        got = _clean_fields(raw, missing)
        facts.update(got)
        still = [f for f in missing if f not in got]
        # 没抽全就留在这一步（cursor 不动），下一轮再问
        return Command(goto="step", update={"facts": facts, "model_calls": 1, **({} if still else nxt),
                                            "trail": [f"{s['id']}：坐席答「{answer}」→ 抽出 {got}" + (f"，还缺 {still}" if still else "")]})

    if s["kind"] == "approve":
        amount = float(facts["amount"])
        tier = next(t for t in s["tiers"] if amount <= t.get("max", float("inf")))
        if tier["level"] == "auto":
            facts["approver"] = "System"
            return Command(goto="step", update={**nxt, "facts": facts, "trail": [f"{s['id']}：{amount:g} USD 在自动审批档，审批人 System"]})
        decision = interrupt({"kind": "approve", "step": s["id"], "level": tier["level"],
                              "summary": {k: facts.get(k) for k in ("order_id", "amount", "comp_type", "reason", "bank_account")}})
        if str(decision).strip().lower().startswith("approve"):
            facts["approver"] = tier["level"]
            return Command(goto="step", update={**nxt, "facts": facts, "trail": [f"{s['id']}：{tier['level']} 审批通过"]})
        back = s["on_reject"]
        for f in back.get("reset", []):
            facts.pop(f, None)
        return Command(goto="step", update={"cursor": step_index(sop, back["goto"]), "facts": facts,
                                            "trail": [f"{s['id']}：{tier['level']} 拒绝——{decision}。回退到 {back['goto']}，清掉 {back.get('reset')}"]})

    raise ValueError(f"不认识的步骤类型 {s['kind']}")


def _clean_fields(raw: dict, wanted: list[str]) -> dict:
    """json_mode 不保证类型，能校验的都在代码里校验。"""
    out = {}
    for f in wanted:
        v = raw.get(f)
        if v in (None, "", "null"):
            continue
        if f == "amount":
            try:
                v = float(v)
                v = int(v) if v.is_integer() else v
            except (TypeError, ValueError):
                continue
        if f == "comp_type" and v not in ("cash", "credit"):
            continue
        out[f] = v
    return out


# ---------- 模型出场三：写小结 ----------

def finish(state: SopState) -> dict:
    run = state["trail"]
    start = max(i for i, line in enumerate(run) if line.startswith("── ")) if any(l.startswith("── ") for l in run) else 0
    text = llm.invoke(SUMMARY.format(trail="\n".join(run[start:]))).content
    return {"messages": [AIMessage(text)], "model_calls": 1}


def build_graph(checkpointer):
    g = StateGraph(SopState)
    g.add_node("plan", plan)
    g.add_node("qa", qa)
    g.add_node("step", step)
    g.add_node("finish", finish)
    g.add_edge(START, "plan")
    g.add_edge("qa", END)
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)
