"""跑例子 5（user_id 认人，thread_id 认对话）：
    uv run python -m ex05_memory_agent.main zhao t1 "我下周带我妈去东京，她 78 岁腿脚不好，我们都不吃牛肉"
    uv run python -m ex05_memory_agent.main zhao t1 "对了我妈现在改吃素了"
    uv run python -m ex05_memory_agent.main zhao t2 "帮我推荐一下东京的餐厅"        # 新对话，没有历史
    uv run python -m ex05_memory_agent.main --memories zhao                          # 看记忆库

记忆在 Store 里按 ("memories", user_id) 存，一条一个 key；对话在 checkpointer 里按 thread_id 存。
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from ex05_memory_agent.embed import index_config
from ex05_memory_agent.graph import build_graph

DATA = Path(__file__).parent / "data"
CHECKPOINT_DB = DATA / "checkpoints.sqlite"
MEMORY_DB = DATA / "memories.sqlite"


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(1)
    DATA.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver, \
            SqliteStore.from_conn_string(str(MEMORY_DB), index=index_config()) as store:
        store.setup()
        if args[0] == "--memories":
            items = store.search(("memories", args[1]), limit=50)
            print(f"user {args[1]} 的记忆库：{len(items)} 条")
            for it in items:
                print(f"  [{it.key}] {it.value['content']}（{it.value['context']}，{it.value['updated']}）")
            return

        user_id, thread_id, text = args[0], args[1], " ".join(args[2:])
        graph = build_graph(saver, store)
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        print(f"[{user_id}/{thread_id}] 客人：{text}")
        for update in graph.stream({"messages": [HumanMessage(text)]}, config=config, stream_mode="updates"):
            for node, changed in update.items():
                for msg in changed.get("messages", []):
                    if node == "call_model" and msg.tool_calls:
                        for c in msg.tool_calls:
                            print(f"  调用 upsert_memory({c['args']})")
                    elif node == "store_memory":
                        print(f"  工具返回：{msg.content}")
                    elif node == "call_model":
                        print(f"  助理：{msg.content}")


if __name__ == "__main__":
    main()
