"""跑第 5 期：
    uv run python -m ep05.main t1 "帮我取消 KL-315"
    uv run python -m ep05.main --resume t1 approve   # 人工同意
    uv run python -m ep05.main --resume t1 reject    # 人工拒绝
    uv run python -m ep05.main --history t1
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ep05.graph import build_graph

DB = Path(__file__).parent / "data" / "checkpoints.sqlite"


def print_stream(stream) -> None:
    for update in stream:
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            print(f"[需要人工确认] {info.value}")
            print("  用 --resume <thread_id> approve 或 reject 继续")
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


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(1)

    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(saver)

        if args[0] == "--history":
            show_history(graph, {"configurable": {"thread_id": args[1]}})
            return

        if args[0] == "--resume":
            if len(args) < 3:
                print(__doc__)
                raise SystemExit(1)
            thread_id, decision = args[1], args[2]
            config = {"configurable": {"thread_id": thread_id}}
            print_stream(graph.stream(Command(resume=decision), config=config, stream_mode="updates"))
            return

        thread_id, question = args[0], " ".join(args[1:])
        config = {"configurable": {"thread_id": thread_id}}
        state = {"messages": [HumanMessage(question)]}
        print_stream(graph.stream(state, config=config, stream_mode="updates"))


if __name__ == "__main__":
    main()
