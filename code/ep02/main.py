"""跑第 2 期：
    uv run python -m ep02.main SKU-1001 "我想把日期改到下周六可以吗"
    uv run python -m ep02.main --steps SKU-1001 "我想把日期改到下周六可以吗"
    uv run python -m ep02.main --show-graph
"""

import sys

from ep02.graph import graph


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--show-graph":
        print(graph.get_graph().draw_mermaid())
        return
    steps = bool(args) and args[0] == "--steps"
    if steps:
        args = args[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(1)
    state = {"product_id": args[0], "question": " ".join(args[1:]), "trace": []}

    if steps:
        # stream_mode="updates"：每个节点跑完，吐出它返回的那一小块更新
        for update in graph.stream(state, stream_mode="updates"):
            for node, changed in update.items():
                print(f"[{node}] 写入 {', '.join(changed)}")
        return

    # 一次跑完拿最终状态。version="v2" 返回的是结构化结果，最终状态在 .value 里
    result = graph.invoke(state, version="v2")
    print("trace:", " | ".join(result.value["trace"]))
    print("草稿:", result.value["draft"])


if __name__ == "__main__":
    main()
