"""第 5 期：节点和边跟第 3-4 期一模一样，改动全在 tools.py 里的 cancel_order。"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from common.llm import chat_model
from ep05.prompts import system_prompt
from ep05.state import AgentState
from ep05.tools import TOOLS

llm = chat_model().bind_tools(TOOLS)


def agent(state: AgentState) -> dict:
    reply = llm.invoke([SystemMessage(system_prompt()), *state["messages"]])
    return {"messages": [reply]}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
