"""跑第 3 期：
    uv run python -m ep03.main "订单 KL-778 能改到下周六吗"
    uv run python -m ep03.main --limit 2 "订单 KL-778 能改到下周六吗"
    uv run python -m ep03.main --show-graph
"""

import sys

from langchain_core.messages import HumanMessage

from ep03.graph import graph


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--show-graph":
        print(graph.get_graph().draw_mermaid())
        return
    limit = 20  # 一次对话最多走多少步，防止模型和工具之间来回打转
    if args and args[0] == "--limit":
        limit, args = int(args[1]), args[2:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    state = {"messages": [HumanMessage(" ".join(args))]}
    for update in graph.stream(state, stream_mode="updates", config={"recursion_limit": limit}):
        for node, changed in update.items():
            for msg in changed["messages"]:
                if node == "agent" and msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"[agent] 要调 {call['name']}({call['args']})")
                elif node == "agent":
                    print(f"[agent] 回答：{msg.content}")
                else:
                    print(f"[tools] {msg.name} 返回：{msg.content}")


if __name__ == "__main__":
    main()
