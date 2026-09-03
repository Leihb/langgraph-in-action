"""第 10 期新增一条支线：`tools` 节点之后不再是一条固定边，而是一次路由判断
（`route_after_tools`）。平时（没人调 `check_orders`）它照旧直接回 `agent`，
跟第 3-9 期一模一样。一旦 `check_orders` 把 `order_ids` 写进了状态，它改用
`Send` 给 `lookup_order` 这个节点连发 N 份任务——这是"多 agent 并行"在
LangGraph 里的样子：扇出几份，由**图的结构**决定，不是模型当场调了几次工具。

`lookup_order` 每次只领到一个订单号，转手交给 `subagent.py` 那个独立编译的
子图去查，互不看见对方——这是"子图"：一个完整的、能单独跑的小 agent，被
父图当一个节点使用。所有 `lookup_order` 分支跑完，`aggregate` 把结果拼成
一条新消息追加进历史，再回到 `agent`，模型这才第一次看到结果。

`check_orders` 那次工具调用本身，在扇出的第一时间就已经拿到一条 `ToolMessage`
应付过去了（"结果在后面那条消息里"）——OpenAI 协议要求每次工具调用必须有
一条配对的工具结果，不能等扇出跑完才补，`aggregate` 加的是另一条独立的消息，
不是这条工具结果本身。"""

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Send

from common.llm import chat_model
from ep14.prompts import system_prompt
from ep14.state import AgentState
from ep14.subagent import build_order_subgraph

order_subgraph = build_order_subgraph()


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def route_after_tools(state: AgentState) -> str | list[Send]:
    order_ids = state.get("order_ids")
    if not order_ids:
        return "agent"
    aspect = state.get("check_aspect", "")
    return [Send("lookup_order", {"order_id": oid, "check_aspect": aspect}) for oid in order_ids]


async def lookup_order(state: AgentState) -> dict:
    order_id = state["order_id"]
    aspect = state["check_aspect"]
    started = time.monotonic()
    print(f"[lookup_order] {order_id} 开始（t={started:.2f}）")
    result = await order_subgraph.ainvoke(
        {"messages": [SystemMessage(f"只回答这一句：订单 {order_id}，{aspect}")]}
    )
    print(f"[lookup_order] {order_id} 结束（耗时 {time.monotonic() - started:.2f}s）")
    return {"order_reports": {order_id: result["messages"][-1].content}}


def aggregate(state: AgentState) -> dict:
    reports = state.get("order_reports", {})
    lines = [f"{oid}：{reports.get(oid, '没查到结果')}" for oid in state.get("order_ids", [])]
    note = HumanMessage("[并行核对结果]\n" + "\n".join(lines))
    return {"messages": [note], "order_ids": []}


def build_graph(checkpointer, store, tools: list):
    llm = chat_model().bind_tools(tools)

    def agent(state: AgentState, config: RunnableConfig, runtime: Runtime) -> dict:
        user_id = config["configurable"]["user_id"]
        item = runtime.store.get((user_id, "memory"), "note")
        note = item.value["text"] if item else None
        reply = llm.invoke([SystemMessage(system_prompt(note)), *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("lookup_order", lookup_order)
    builder.add_node("aggregate", aggregate)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", route_after_tools, ["agent", "lookup_order"])
    builder.add_edge("lookup_order", "aggregate")
    builder.add_edge("aggregate", "agent")
    return builder.compile(checkpointer=checkpointer, store=store)
