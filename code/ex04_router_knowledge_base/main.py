"""跑例子 4：
    uv run python -m ex04_router_knowledge_base.main "客人机场大巴票过了出发时间还能退吗，以前有没有类似处理？"
    uv run python -m ex04_router_knowledge_base.main --show-graph

没有 checkpointer：一问一答，不需要跨轮记忆。
"""

import sys
import time

from ex04_router_knowledge_base.graph import graph


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    if args[0] == "--show-graph":
        print(graph.get_graph().draw_mermaid())
        return
    question = " ".join(args)
    print(f"=== {question} ===")
    t0 = time.monotonic()
    final = None
    for update in graph.stream({"question": question, "results": []}, stream_mode="updates"):
        for node, changed in update.items():
            if node == "classify":
                for c in changed["classifications"]:
                    print(f"  路由 -> {c['source']}：{c['query']}")
                if not changed["classifications"]:
                    print("  路由 -> 没有相关来源")
            elif node == "synthesize":
                final = changed["final_answer"]
            else:  # 某个来源的汇报
                (r,) = changed["results"]
                print(f"  [{r['source']}] 汇报：{r['result'][:160].replace(chr(10), ' ')}…")
    print(f"\n（总耗时 {time.monotonic() - t0:.1f}s）")
    print(f"\n回答：\n{final}")


if __name__ == "__main__":
    main()
