"""第 3 期：工具调用与条件边。模型决定走哪条边。"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from common.llm import chat_model
from ep03.prompts import system_prompt
from ep03.state import AgentState
from ep03.tools import TOOLS

# bind_tools 把工具的名字、参数、docstring 变成模型能看懂的声明，随每次请求一起发出去
llm = chat_model().bind_tools(TOOLS)


def agent(state: AgentState) -> dict:
    """节点一：模型看完整对话，决定是回答还是调工具。"""
    reply = llm.invoke([SystemMessage(system_prompt()), *state["messages"]])
    return {"messages": [reply]}


def route(state: AgentState) -> str:
    """条件边：模型最后一条消息里有没有工具调用请求？有就去 tools，没有就结束。"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))  # 节点二：执行工具，把结果写回对话
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")  # 工具结果回到模型手里，循环在这里闭合
    return builder.compile()


graph = build_graph()
