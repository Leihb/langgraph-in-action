"""方案二：检索当节点。图头上多一个 `retrieve` 节点，每一轮对话开始都会跑，
不经过模型判断——不管这句话用不用得上 FAQ，都先查一次。`search_faq` 不在
这个方案的工具列表里，检索结果直接写进 `state["retrieved"]`，`agent` 节点
读出来拼进系统提示词。"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from common.llm import chat_model
from ep08 import retrieval
from ep08.prompts import system_prompt
from ep08.state import AgentState


def retrieve(state: AgentState) -> dict:
    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    hits = retrieval.search_faq(last_human.content, top_k=2)
    formatted = "\n".join(f"- {h['question']}：{h['answer']}（相似度 {h['score']:.2f}）" for h in hits)
    return {"retrieved": formatted}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(checkpointer, store, tools: list):
    llm = chat_model().bind_tools(tools)

    def agent(state: AgentState, config: RunnableConfig, runtime: Runtime) -> dict:
        user_id = config["configurable"]["user_id"]
        item = runtime.store.get((user_id, "memory"), "note")
        note = item.value["text"] if item else None
        reply = llm.invoke(
            [SystemMessage(system_prompt(note, state.get("retrieved"))), *state["messages"]]
        )
        return {"messages": [reply]}

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    # 工具循环回到 agent，不重新经过 retrieve——一轮对话只检索一次，
    # 检索的是这一轮最新那句人话，不是每次工具调用都重新查。
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, store=store)
