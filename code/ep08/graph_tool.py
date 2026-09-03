"""方案一：检索当工具。节点和边的结构跟第 6-7 期完全一样——`search_faq`
只是 TOOLS 列表里多的一项，agent 自己判断要不要调它。"""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from common.llm import chat_model
from ep08.prompts import system_prompt
from ep08.state import AgentState


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


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
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, store=store)
