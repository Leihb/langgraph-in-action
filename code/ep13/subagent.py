"""order_subgraph：一个独立编译的迷你 agent，只有两个工具（get_order、get_policy），
只看得见分给它的这一句任务，看不到父对话的历史，也看不到别的子任务在查什么。
这跟上册练习 19 的 sub_agent 是同一个设计：隔离的上下文，一次性的任务，办完就退出。

不带 checkpointer——每次 `.ainvoke()` 都是一次从零开始、跑完就丢的运行，
不需要跨轮记住什么。"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from common.llm import chat_model
from ep13.tools import get_order, get_policy

SUB_TOOLS = [get_order, get_policy]
SUB_PROMPT = "你是订单查询助手，只回答被问到的那一句话，一句话作答，不客套。"


class SubState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _route(state: SubState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_order_subgraph():
    llm = chat_model().bind_tools(SUB_TOOLS)

    def agent(state: SubState) -> dict:
        reply = llm.invoke([SystemMessage(SUB_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(SubState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(SUB_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()
