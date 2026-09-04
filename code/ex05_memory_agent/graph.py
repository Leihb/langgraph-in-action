"""例子 5：长期记忆 agent。langchain-ai/memory-agent 仓库那个例子的重做。

第 6 期的记忆是一人一条笔记，模型整条覆盖；这一篇升级成一人一个记忆库：一条记忆一个
key，新增和修改分开，每次调模型前按当前话题把最相关的几条捞出来放进提示词。

    call_model（检索记忆 → 拼进系统提示词 → 调模型）
      ├─ 有 tool_calls → store_memory（ToolNode，执行 upsert_memory）→ 回到 call_model
      └─ 没有 → 结束

存回去之后回到 call_model 而不是直接结束——官方代码里那行注释说的："看模型，你可能想
让它先存再回答"。这一篇就这么做，模型存完记忆还要对客人说一句话。
"""

from datetime import date

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from common.llm import chat_model
from ex05_memory_agent.prompts import SYSTEM
from ex05_memory_agent.tools import upsert_memory

llm = chat_model().bind_tools([upsert_memory])


def recall(store, user_id: str, query: str, limit: int = 6) -> list:
    """按当前话题捞记忆。Store 建了索引就是语义检索（query 生效）；没建索引 query 会被
    忽略，按写入顺序返回——两种情况都走这一个调用。"""
    return store.search(("memories", user_id), query=query, limit=limit)


def call_model(state: MessagesState, config: RunnableConfig, runtime: Runtime) -> dict:
    # 节点签名里声明 config 和 runtime，LangGraph 会把这次运行的配置和运行时（含 store）注入进来
    user_id = config["configurable"]["user_id"]
    # 用最近三条消息当检索词：客人这次在聊什么，就捞跟它相关的记忆
    query = " ".join(str(m.content) for m in state["messages"][-3:] if m.content)
    items = recall(runtime.store, user_id, query)
    formatted = "\n".join(f"[{it.key}] {it.value['content']}（{it.value['context']}）" for it in items) or "（还没有任何记忆）"
    print(f"  [recall] 召回 {len(items)} 条：{[it.key for it in items]}")
    reply = llm.invoke([SystemMessage(SYSTEM.format(today=date.today().isoformat(), memories=formatted)), *state["messages"]])
    return {"messages": [reply]}


def route(state: MessagesState) -> str:
    return "store_memory" if state["messages"][-1].tool_calls else END


def build_graph(checkpointer, store):
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("store_memory", ToolNode([upsert_memory]))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route, ["store_memory", END])
    builder.add_edge("store_memory", "call_model")
    return builder.compile(checkpointer=checkpointer, store=store)
