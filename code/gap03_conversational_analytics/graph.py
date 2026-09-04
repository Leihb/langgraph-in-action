"""空白 3：对话式数据分析——SQL、图表、多轮细化。

    understand（模型：SQL + 图表偏好）→ check（代码）→ run（代码）→ chart（代码：选图、画图）→ narrate（模型）
                     ▲                   │ 校验/执行报错 ≤3 次
                     └───────────────────┘
    mode=chart_only：跳过 check/run，拿上一轮的结果直接 chart。
    mode=cannot：直接 END。

跟例子 2 的差别：①多轮——state 里留着前几轮的问题、SQL 和结果，用户说"拆成按品类"，模型在上一轮
SQL 上改；②图表——结果不只是一段话，是一份结构化的图表规格 + 终端里的字符图，选什么图由代码
按结果的形状决定；③没有执行前审批——只读连接 + SELECT-only + EXPLAIN 三道保险留着，审批那一步
例子 2 讲过，业务用户问数的场景等不起。
"""

import json
import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from common.llm import chat_model
from gap03_conversational_analytics import charts, db
from gap03_conversational_analytics.prompts import HISTORY_ITEM, NARRATE, UNDERSTAND

MAX_ATTEMPTS = 3
llm = chat_model()
json_llm = chat_model().with_structured_output(None, method="json_mode")


class Turn(TypedDict, total=False):
    question: str
    sql: str | None
    columns: list[str]
    rows: list[list]
    chart: dict           # 图表规格
    narrative: str


class AnalyticsState(TypedDict):
    question: str
    turns: Annotated[list[Turn], operator.add]   # 已完成的轮次
    mode: str
    sql: str | None
    chart_hint: str | None
    title: str
    reason: str
    error: str
    attempts: int
    columns: list[str]
    rows: list[list]
    chart: dict
    caveat: str
    rendered: str
    narrative: str
    trail: Annotated[list[str], operator.add]
    model_calls: Annotated[int, operator.add]


def understand(state: AnalyticsState) -> Command[Literal["check", "chart", "__end__"]]:
    history = ""
    if state.get("turns"):
        recent = state["turns"][-3:]
        history = "前几轮的对话（最近的在最后）：\n" + "".join(
            HISTORY_ITEM.format(i=i + 1, question=t["question"], sql=t.get("sql"), n=len(t.get("rows", [])),
                                columns=", ".join(t.get("columns", []))) for i, t in enumerate(recent)) + "\n"
    if state.get("error"):
        history += f"你上一版 SQL 有问题：{state['error']}。请改正后重写。\n\n"
    out = json_llm.invoke(UNDERSTAND.format(today=db.TODAY.isoformat(), schema=db.load_schema(),
                                            history=history, question=state["question"]))
    mode = out.get("mode") if out.get("mode") in ("query", "chart_only", "cannot") else "cannot"
    chart = out.get("chart") or {}
    attempts = state.get("attempts", 0) + 1
    update = {"mode": mode, "sql": out.get("sql"), "chart_hint": chart.get("type"), "title": str(chart.get("title") or state["question"]),
              "reason": str(out.get("reason", "")), "attempts": attempts, "error": "", "model_calls": 1}
    if mode == "cannot" or (mode == "query" and not out.get("sql")):
        return Command(goto=END, update={**update, "mode": "cannot", "narrative": f"这个问题查不了：{update['reason']}",
                                         "trail": [f"understand #{attempts} → cannot：{update['reason']}"]})
    if mode == "chart_only":
        if not state.get("turns"):
            return Command(goto=END, update={**update, "mode": "cannot", "narrative": "还没有上一轮结果可以换图",
                                             "trail": [f"understand #{attempts} → chart_only 但没有上一轮"]})
        last = state["turns"][-1]
        return Command(goto="chart", update={**update, "sql": last.get("sql"), "columns": last["columns"], "rows": last["rows"],
                                             "trail": [f"understand #{attempts} → 只换图：{chart.get('type')}，数据沿用上一轮"]})
    return Command(goto="check", update={**update, "trail": [f"understand #{attempts} → {update['sql'].strip()[:90]}"]})


def check(state: AnalyticsState) -> Command[Literal["run", "understand", "__end__"]]:
    try:
        db.check_query(state["sql"])
    except db.QueryRejected as e:
        return _retry_or_giveup(state, f"校验没过：{e}")
    return Command(goto="run", update={"trail": ["check → 通过"]})


def run(state: AnalyticsState) -> Command[Literal["chart", "understand", "__end__"]]:
    try:
        columns, rows = db.run_query(state["sql"])
    except Exception as e:  # noqa: BLE001
        return _retry_or_giveup(state, f"执行报错：{e}")
    return Command(goto="chart", update={"columns": columns, "rows": rows, "trail": [f"run → {len(rows)} 行 × {len(columns)} 列"]})


def _retry_or_giveup(state, msg: str) -> Command:
    if state["attempts"] >= MAX_ATTEMPTS:
        return Command(goto=END, update={"mode": "cannot", "narrative": f"改了 {MAX_ATTEMPTS} 次还没通过：{msg}",
                                         "trail": [f"{msg}（第 {state['attempts']} 次，放弃）"]})
    return Command(goto="understand", update={"error": msg, "trail": [f"{msg} → 回去改"]})


def chart(state: AnalyticsState) -> dict:
    kind, columns, rows, note = charts.choose(state["columns"], state["rows"], state.get("chart_hint"))
    caveat = charts.partial_period_caveat(columns, rows, db.TODAY)
    spec = charts.spec(kind, columns, rows, state["title"])
    if caveat:
        spec["caveat"] = caveat
    rendered = charts.render(kind, columns, rows, state["title"]) + (f"\n   ⚠ {caveat}" if caveat else "")
    return {"chart": spec, "rendered": rendered, "columns": columns, "rows": rows, "caveat": caveat,
            "trail": [f"chart → {kind}（{len(rows)} 行 × {len(columns)} 列）" + (f"，{note}" if note else "") + (f"，标注：{caveat}" if caveat else "")]}


def narrate(state: AnalyticsState) -> dict:
    rows = state["rows"]
    if state["mode"] == "chart_only":
        text = state["turns"][-1].get("narrative", "")
        calls = 0
    else:
        caveat = f"注意：{state['caveat']}\n" if state.get("caveat") else ""
        text = llm.invoke(NARRATE.format(question=state["question"], n=len(rows), columns=", ".join(state["columns"]),
                                         rows="\n".join(" | ".join(str(v) for v in r) for r in rows[:60]), caveat=caveat)).content.strip()
        calls = 1
    turn: Turn = {"question": state["question"], "sql": state["sql"], "columns": state["columns"], "rows": rows,
                  "chart": state["chart"], "narrative": text}
    return {"narrative": text, "turns": [turn], "model_calls": calls, "trail": ["narrate → 两句话"] if calls else []}


def build_graph(checkpointer):
    g = StateGraph(AnalyticsState)
    for name, fn in [("understand", understand), ("check", check), ("run", run), ("chart", chart), ("narrate", narrate)]:
        g.add_node(name, fn)
    g.add_edge(START, "understand")
    g.add_edge("chart", "narrate")
    g.add_edge("narrate", END)
    return g.compile(checkpointer=checkpointer)
