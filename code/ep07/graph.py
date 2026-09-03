"""第 7 期：节点和边的结构跟第 6 期一样。唯一的结构性改动是 `build_graph`
多了一个 `tools` 参数——MCP 工具是运行时问服务器要来的，不是 import 时就
定死的列表，所以 `llm.bind_tools` 挪进了函数内部。"""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from common.llm import chat_model
from ep07.prompts import system_prompt
from ep07.state import AgentState


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
