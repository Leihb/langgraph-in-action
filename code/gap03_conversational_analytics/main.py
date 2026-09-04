"""空白 3：对话式数据分析。

    uv run python -m gap03_conversational_analytics.main a1 "过去 12 个月每月的销售额"
    uv run python -m gap03_conversational_analytics.main a1 "拆成按品类"
    uv run python -m gap03_conversational_analytics.main a1 "只看日本"
    uv run python -m gap03_conversational_analytics.main a1 "换成柱状图"
    uv run python -m gap03_conversational_analytics.main --spec a1        # 看最后一轮的图表规格 JSON
    uv run python -m gap03_conversational_analytics.main --history a1

一个 thread 一段对话。每轮打印：执行记录、字符图、两句话、这轮调了几次模型。加 --sql 打印 SQL。
"""

import json
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from gap03_conversational_analytics.graph import build_graph

DB = Path(__file__).parent / "data" / "checkpoints.sqlite"


def main() -> None:
    args = sys.argv[1:]
    show_sql = "--sql" in args
    args = [a for a in args if a != "--sql"]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(1)
    DB.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(saver)
        if args[0] in ("--spec", "--history"):
            v = graph.get_state({"configurable": {"thread_id": args[1]}}).values
            turns = v.get("turns", [])
            if args[0] == "--spec":
                print(json.dumps(turns[-1]["chart"], ensure_ascii=False, indent=1)[:1500] if turns else "还没有结果")
            else:
                for i, t in enumerate(turns, 1):
                    print(f"  {i}. {t['question']}\n     SQL: {(t.get('sql') or '').strip()[:100]}\n     {t['chart']['type']}，{len(t.get('rows', []))} 行")
            return
        thread_id, question = args[0], " ".join(args[1:])
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        before = (graph.get_state(config).values or {}).get("model_calls", 0)
        print(f"[{thread_id}] 用户：{question}")
        for update in graph.stream({"question": question}, config=config, stream_mode="updates"):
            for _node, changed in update.items():
                if isinstance(changed, dict):
                    for line in changed.get("trail", []):
                        print(f"  {line}")
        v = graph.get_state(config).values
        if show_sql and v.get("sql"):
            print("  SQL:", " ".join(v["sql"].split()))
        if v.get("rendered") and v.get("mode") != "cannot":
            print("\n" + "\n".join("  " + l for l in v["rendered"].splitlines()) + "\n")
        print(f"  {v.get('narrative', '')}")
        print(f"  （这轮模型调用 {v.get('model_calls', 0) - before} 次）")


if __name__ == "__main__":
    main()
