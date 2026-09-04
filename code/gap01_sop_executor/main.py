"""空白 1：按 SOP 一步步执行的坐席助手。

    uv run python -m gap01_sop_executor.main t1 "给 KL-778 补 80 美元现金，商户漏发接送"
    uv run python -m gap01_sop_executor.main t1 "中国银行 6222 0000 1234，户名王女士"   # 回答上一步的提问
    uv run python -m gap01_sop_executor.main t1 approve                                  # 审批人批准（或 "reject 理由"）
    uv run python -m gap01_sop_executor.main --state t1                                  # 看 facts 和执行记录

thread 停在 interrupt 上时，发来的消息自动当作回答（跟例子 7 的服务一样）。
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from gap01_sop_executor.graph import build_graph

DB = Path(__file__).parent / "data" / "checkpoints.sqlite"


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(1)
    DB.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(saver)
        if args[0] == "--state":
            v = graph.get_state({"configurable": {"thread_id": args[1]}}).values
            print(f"  sop={v.get('sop')} cursor={v.get('cursor')} outcome={v.get('outcome')} model_calls={v.get('model_calls')}")
            print(f"  facts: { {k: val for k, val in (v.get('facts') or {}).items() if k != 'order'} }")
            for line in v.get("trail", []):
                print(f"  {line}")
            return

        thread_id, text = args[0], " ".join(args[1:])
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
        snapshot = graph.get_state(config)
        interrupted = any(getattr(t, "interrupts", None) for t in snapshot.tasks)
        before = snapshot.values.get("model_calls", 0) if snapshot.values else 0
        run_input = Command(resume=text) if interrupted else {"messages": [HumanMessage(text)]}
        print(f"[{thread_id}] 坐席：{text}" + ("   （作为对上一步提问的回答）" if interrupted else ""))
        for update in graph.stream(run_input, config=config, stream_mode="updates"):
            for node, changed in update.items():
                if node == "__interrupt__":
                    for it in changed:
                        v = it.value
                        if v["kind"] == "ask":
                            print(f"  ⏸ 问坐席：{v['prompt']}（缺 {v['missing']}）")
                        else:
                            print(f"  ⏸ 等 {v['level']} 审批：{v['summary']}")
                    continue
                if isinstance(changed, dict):
                    for line in changed.get("trail", []):
                        print(f"  {line}")
        v = graph.get_state(config).values
        if v["messages"] and v["messages"][-1].type == "ai" and not interrupted_now(graph, config):
            print(f"  助手：{v['messages'][-1].content}")
        print(f"  （这轮模型调用 {v.get('model_calls', 0) - before} 次，累计 {v.get('model_calls', 0)}）")


def interrupted_now(graph, config) -> bool:
    return any(getattr(t, "interrupts", None) for t in graph.get_state(config).tasks)


if __name__ == "__main__":
    main()
