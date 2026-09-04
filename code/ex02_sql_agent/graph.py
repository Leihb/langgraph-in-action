"""例子 2：SQL 问数——执行前停下来批准。官方 "Build a custom SQL agent" 教程的重做。

官方那张图：list_tables → 模型被强制调 get_schema → generate_query（模型，绑着
run_query 工具）→ check_query（再让模型复查一遍 SQL）→ run_query（ToolNode）→ 回到
generate_query。六个节点里模型出现三次，两次是被 tool_choice 强制调工具。

这一篇的图没有 ToolNode，模型不"调工具"，SQL 是它按结构吐出来的一个字段：

    load_schema（代码）→ generate_query（模型）→ check_query（代码：规则 + 数据库 EXPLAIN）
      → approve_query（interrupt）→ run_query（代码，只读连接）→ answer（模型）

模型出现两次，都在填空。校验交给数据库自己做（EXPLAIN QUERY PLAN 会报语法错和不存在
的表名列名），比再叫一次模型"复查常见错误"准。改成这样有一个现实原因：第 1 个例子
真机撞见 DeepSeek 的思考模式拒绝强制 tool_choice——官方那两处强制调用在这台端点上
根本跑不起来。
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common.llm import chat_model
from ex02_sql_agent import db
from ex02_sql_agent.prompts import ANSWER, CANNOT, GENERATE
from ex02_sql_agent.state import SqlState

MAX_ATTEMPTS = 3

llm = chat_model()
# 结构化输出走 json_mode：第 1 个例子试过，这台端点另外两种协议都是 400
generator = llm.with_structured_output({"title": "Query", "type": "object",
                                        "properties": {"sql": {"type": ["string", "null"]}, "reason": {"type": "string"}},
                                        "required": ["sql", "reason"]}, method="json_mode")


def load_schema(state: SqlState) -> dict:
    return {"schema": db.load_schema(), "attempts": 0, "trace": ["load_schema -> 11 张表"]}


def generate_query(state: SqlState) -> Command[Literal["check_query", "answer"]]:
    feedback = ""
    if state.get("check_error"):
        feedback = f"\n上一版 SQL 没有通过校验，原因：{state['check_error']}。请改正后重写。"
    elif state.get("run_error"):
        feedback = f"\n上一版 SQL 执行报错：{state['run_error']}。请改正后重写。"
    out = generator.invoke(GENERATE.format(schema=state["schema"], question=state["question"], feedback=feedback))
    sql, reason = out.get("sql"), str(out.get("reason", ""))
    attempts = state.get("attempts", 0) + 1
    update = {"sql": sql, "reason": reason, "attempts": attempts, "check_error": "", "run_error": ""}
    if not sql:
        # 模型自己说查不了——不硬编 SQL，直接去解释
        return Command(update={**update, "trace": [f"generate #{attempts} -> 模型判断查不了：{reason}"]}, goto="answer")
    return Command(update={**update, "trace": [f"generate #{attempts} -> {sql.strip()[:80]}"]}, goto="check_query")


def check_query(state: SqlState) -> Command[Literal["approve_query", "generate_query", "answer"]]:
    try:
        db.check_query(state["sql"])
    except db.QueryRejected as e:
        if state["attempts"] >= MAX_ATTEMPTS:
            return Command(update={"check_error": str(e), "sql": None,
                                   "reason": f"改了 {MAX_ATTEMPTS} 次仍未通过校验：{e}",
                                   "trace": [f"check -> 拒绝（第 {state['attempts']} 次，放弃）：{e}"]}, goto="answer")
        return Command(update={"check_error": str(e), "trace": [f"check -> 拒绝，回去改：{e}"]}, goto="generate_query")
    return Command(update={"trace": ["check -> 通过"]}, goto="approve_query")


def approve_query(state: SqlState) -> Command[Literal["run_query", "answer"]]:
    """interrupt 放第一行。给人看的是 SQL 本身和模型的一句说明——批的是"这条查询
    可以在库上跑"，跟第 5 期批"这个订单可以取消"是同一个机制。"""
    decision = interrupt({
        "question": state["question"],
        "sql": state["sql"],
        "reason": state["reason"],
        "options": ["accept", "edit:<改好的 SQL>", "reject"],
    })
    if decision == "accept":
        return Command(update={"approval": "accept", "trace": ["approve -> accept"]}, goto="run_query")
    if isinstance(decision, str) and decision.startswith("edit:"):
        new_sql = decision[5:].strip()
        # 人改过的 SQL 也要过一遍同样的校验——人也会写错
        try:
            db.check_query(new_sql)
        except db.QueryRejected as e:
            return Command(update={"approval": "edit", "sql": None, "reason": f"人工改写的 SQL 没通过校验：{e}",
                                   "trace": [f"approve -> edit 但校验没过：{e}"]}, goto="answer")
        return Command(update={"approval": "edit", "sql": new_sql, "trace": ["approve -> edit"]}, goto="run_query")
    return Command(update={"approval": "reject", "sql": None, "reason": "人工拒绝执行这条查询",
                           "trace": ["approve -> reject"]}, goto="answer")


def run_query(state: SqlState) -> Command[Literal["answer", "generate_query"]]:
    try:
        columns, rows = db.run_query(state["sql"])
    except Exception as e:  # noqa: BLE001 — 执行期错误一律回给模型改，或放弃
        if state["attempts"] >= MAX_ATTEMPTS:
            return Command(update={"run_error": str(e), "sql": None, "reason": f"执行报错且已改 {MAX_ATTEMPTS} 次：{e}",
                                   "trace": [f"run -> 报错（放弃）：{e}"]}, goto="answer")
        return Command(update={"run_error": str(e), "trace": [f"run -> 报错，回去改：{e}"]}, goto="generate_query")
    return Command(update={"columns": columns, "rows": rows, "trace": [f"run -> {len(rows)} 行"]}, goto="answer")


def answer(state: SqlState) -> dict:
    if not state.get("sql"):
        text = llm.invoke(CANNOT.format(question=state["question"], reason=state.get("reason", ""))).content
        return {"answer": text.strip(), "trace": ["answer -> 解释为什么查不了"]}
    rows = state.get("rows", [])
    text = llm.invoke(ANSWER.format(
        question=state["question"], sql=state["sql"], columns=", ".join(state.get("columns", [])),
        n=len(rows), capped="，已截断" if len(rows) >= db.ROW_CAP else "",
        rows="\n".join(" | ".join(str(v) for v in r) for r in rows),
    )).content
    return {"answer": text.strip(), "trace": ["answer -> 组织回答"]}


def build_graph(checkpointer):
    builder = StateGraph(SqlState)
    for name, fn in [("load_schema", load_schema), ("generate_query", generate_query), ("check_query", check_query),
                     ("approve_query", approve_query), ("run_query", run_query), ("answer", answer)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "load_schema")
    builder.add_edge("load_schema", "generate_query")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)
