"""第 6 期：节点和边跟第 3-5 期一样，`agent` 多了两个注入参数用来读跨会话笔记。"""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from common.llm import chat_model
from ep06.prompts import system_prompt
from ep06.state import AgentState
from ep06.tools import TOOLS

llm = chat_model().bind_tools(TOOLS)


def agent(state: AgentState, config: RunnableConfig, runtime: Runtime) -> dict:
    # config 给 thread_id/user_id 这类"键"，runtime.store 是真正存笔记的地方，
    # 两者分开传是因为 Runtime 本身不带 config（官方注释原话）。
    user_id = config["configurable"]["user_id"]
    item = runtime.store.get((user_id, "memory"), "note")
    note = item.value["text"] if item else None
    reply = llm.invoke([SystemMessage(system_prompt(note)), *state["messages"]])
    return {"messages": [reply]}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(checkpointer, store):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    # 两个存储各管各的：checkpointer 记"这场对话说了什么"，store 记"这个用户身上
    # 值得跨会话保留的东西"。
    return builder.compile(checkpointer=checkpointer, store=store)
