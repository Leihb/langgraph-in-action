"""第 4 期：记住对话。节点和边跟第 3 期一模一样，只在 compile 时挂上 checkpointer。"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from common.llm import chat_model
from ep04.prompts import system_prompt
from ep04.state import AgentState
from ep04.tools import TOOLS

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
    # 唯一的改动：checkpointer 在这里挂上。每个节点跑完，状态落一次盘。
    return builder.compile(checkpointer=checkpointer)
