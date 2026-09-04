"""跑例子 3（多轮对话，一个 thread 一位客人）：
    uv run python -m ex03_support_state_machine.main t1 "你好，我的票想改期"
    uv run python -m ex03_support_state_machine.main t1 "订单号 KL-778"
    uv run python -m ex03_support_state_machine.main t1 "我想改到下周"
    uv run python -m ex03_support_state_machine.main --state t1        # 看 state 里的字段

每一轮打印：模型调了什么工具、当前处于哪一步、模型对客人说了什么。
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from ex03_support_state_machine.agent import build_agent

DB = Path(__file__).parent / "data" / "checkpoints.sqlite"
TRACKED = ("current_step", "order_id", "customer", "product_name", "travel_date", "issue_type", "solution")


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(1)
    with SqliteSaver.from_conn_string(str(DB)) as saver:
        agent = build_agent(saver)
        if args[0] == "--state":
            values = agent.get_state({"configurable": {"thread_id": args[1]}}).values
            for k in TRACKED:
                print(f"  {k:<13} {values.get(k)!r}")
            print(f"  messages      {len(values.get('messages', []))} 条")
            return

        thread_id, text = args[0], " ".join(args[1:])
        config = {"configurable": {"thread_id": thread_id}}
        before = agent.get_state(config).values.get("current_step") or "identify"
        print(f"[{thread_id}] 客人：{text}   （当前步骤：{before}）")
        for update in agent.stream({"messages": [HumanMessage(text)]}, config=config, stream_mode="updates"):
            for node, changed in update.items():
                parts = changed if isinstance(changed, list) else [changed]
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    for msg in part.get("messages", []):
                        if getattr(msg, "tool_calls", None):
                            for c in msg.tool_calls:
                                print(f"  调用 {c['name']}({c['args']})")
                        elif node == "tools":
                            print(f"  工具返回：{str(msg.content)[:80]}")
                    if "current_step" in part:
                        print(f"  → 步骤切到 {part['current_step']}")
        values = agent.get_state(config).values
        print(f"  客服：{values['messages'][-1].content}")
        print(f"  （现在步骤：{values.get('current_step') or 'identify'}）")


if __name__ == "__main__":
    main()
