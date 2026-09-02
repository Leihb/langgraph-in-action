"""跑第 4 期：
    uv run python -m ep04.main t1 "订单 KL-778 能改到下周六吗"
    uv run python -m ep04.main t1 "那退款呢"          # 同一个 thread，接着上次
    uv run python -m ep04.main t2 "那退款呢"          # 换一个 thread，什么都不记得
    uv run python -m ep04.main --history t1           # 看 t1 存了什么
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from ep04.graph import build_graph

DB = Path(__file__).parent / "data" / "checkpoints.sqlite"


def show_history(graph, config) -> None:
    snapshot = graph.get_state(config)
    msgs = snapshot.values.get("messages", [])
    steps = sum(1 for _ in graph.get_state_history(config))
    print(f"thread {config['configurable']['thread_id']}：{len(msgs)} 条消息，{steps} 个 checkpoint")
    for m in msgs:
        kind = type(m).__name__.replace("Message", "")
        text = m.content if m.content else f"tool_calls={[c['name'] for c in m.tool_calls]}"
        print(f"  {kind:<6} {text[:60]}")


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(1)

    # SqliteSaver 把每一步的状态写进一个文件。进程退出再起来，文件还在。
    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(saver)

        if args[0] == "--history":
            show_history(graph, {"configurable": {"thread_id": args[1]}})
            return

        thread_id, question = args[0], " ".join(args[1:])
        # thread_id 是"这是哪一场对话"。同一个 id 进来，就接着上次的状态跑。
        config = {"configurable": {"thread_id": thread_id}}
        state = {"messages": [HumanMessage(question)]}  # 只传新的这一句，历史在 checkpoint 里
        for update in graph.stream(state, config=config, stream_mode="updates"):
            for node, changed in update.items():
                for msg in changed["messages"]:
                    if node == "agent" and msg.tool_calls:
                        for call in msg.tool_calls:
                            print(f"[agent] 要调 {call['name']}({call['args']})")
                    elif node == "agent":
                        print(f"[agent] 回答：{msg.content}")
                    else:
                        print(f"[tools] {msg.name} 返回：{msg.content[:80]}")


if __name__ == "__main__":
    main()
