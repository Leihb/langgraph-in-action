"""跑第 6 期：
    uv run python -m ep06.main wang t1 "以后称呼我王总，不要叫王小姐"
    uv run python -m ep06.main --memory wang
    uv run python -m ep06.main wang t2 "帮我查一下 KL-778 的改期政策"   # 新 thread，同一个人
    uv run python -m ep06.main chen t1 "帮我查一下 KL-901 的退款政策"  # 不同的人
    uv run python -m ep06.main --resume <user_id> <thread_id> approve
    uv run python -m ep06.main --history <thread_id>
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from ep06.graph import build_graph

CHECKPOINT_DB = Path(__file__).parent / "data" / "checkpoints.sqlite"
MEMORY_DB = Path(__file__).parent / "data" / "memory.sqlite"


def print_stream(stream) -> None:
    for update in stream:
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            print(f"[需要人工确认] {info.value}")
            print("  用 --resume <user_id> <thread_id> approve 或 reject 继续")
            continue
        for node, changed in update.items():
            for msg in changed["messages"]:
                if node == "agent" and msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"[agent] 要调 {call['name']}({call['args']})")
                elif node == "agent":
                    print(f"[agent] 回答：{msg.content}")
                else:
                    print(f"[tools] {msg.name} 返回：{msg.content[:80]}")


def show_history(graph, config) -> None:
    snapshot = graph.get_state(config)
    msgs = snapshot.values.get("messages", [])
    print(f"thread {config['configurable']['thread_id']}：{len(msgs)} 条消息，"
          f"下一步待执行节点：{snapshot.next or '(无，对话已结束)'}")
    for m in msgs:
        kind = type(m).__name__.replace("Message", "")
        text = m.content if m.content else f"tool_calls={[c['name'] for c in m.tool_calls]}"
        print(f"  {kind:<6} {text[:60]}")


def show_memory(store, user_id: str) -> None:
    item = store.get((user_id, "memory"), "note")
    print(f"user {user_id} 的笔记：{item.value['text'] if item else '(还没有记过东西)'}")


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 1:
        print(__doc__)
        raise SystemExit(1)

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver, \
            SqliteStore.from_conn_string(str(MEMORY_DB)) as store:
        store.setup()
        graph = build_graph(saver, store)

        if args[0] == "--memory":
            show_memory(store, args[1])
            return

        if args[0] == "--history":
            show_history(graph, {"configurable": {"thread_id": args[1]}})
            return

        if args[0] == "--resume":
            user_id, thread_id, decision = args[1], args[2], args[3]
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            print_stream(graph.stream(Command(resume=decision), config=config, stream_mode="updates"))
            return

        user_id, thread_id, question = args[0], args[1], " ".join(args[2:])
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        state = {"messages": [HumanMessage(question)]}
        print_stream(graph.stream(state, config=config, stream_mode="updates"))


if __name__ == "__main__":
    main()
