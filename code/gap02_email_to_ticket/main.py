"""空白 2：非结构化邮件 → 结构化工单。

    uv run python -m gap02_email_to_ticket.main inbox            # 按收信顺序处理 inbox.json 里的全部邮件
    uv run python -m gap02_email_to_ticket.main c4 file          # 转人工的线程：坐席决定 file / discard
    uv run python -m gap02_email_to_ticket.main --tickets        # 看工单库
    uv run python -m gap02_email_to_ticket.main --outbox         # 看发出去的邮件
    uv run python -m gap02_email_to_ticket.main --reset          # 清空 data/

thread_id = 邮件线程的 conversation_id。停在 human_review 上的线程，再来的邮件先排队（真实系统里
是坐席处理完再进），这里 inbox 模式直接跳过并提示。
"""

import json
import shutil
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from gap02_email_to_ticket import tools
from gap02_email_to_ticket.graph import build_graph

HERE = Path(__file__).parent
DB = HERE / "data" / "checkpoints.sqlite"
INBOX = json.loads((HERE / "inbox.json").read_text(encoding="utf-8"))


def run(graph, config, run_input) -> None:
    for update in graph.stream(run_input, config=config, stream_mode="updates"):
        for node, changed in update.items():
            if node == "__interrupt__":
                for it in changed:
                    v = it.value
                    print(f"  ⏸ 转人工：{v['problems']}\n     草稿 {v['draft']}\n     选项 {v['options']}")
                continue
            if isinstance(changed, dict):
                for line in changed.get("trail", []):
                    print(f"  {line}")


def interrupted(graph, config) -> bool:
    return any(getattr(t, "interrupts", None) for t in graph.get_state(config).tasks)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    if args[0] == "--reset":
        shutil.rmtree(HERE / "data", ignore_errors=True)
        print("data/ 已清空")
        return
    if args[0] == "--tickets":
        for t in tools.all_tickets():
            if t["_op"] == "create":
                print(f"  {t['ticket_no']}  {t['priority']:<7} {t['ticket_type']}/{t['category']:<13} 订单 {t.get('order_id')!s:<7} {t['customer_email']:<28} {t['request']}")
            else:
                print(f"  {t['ticket_no']}  ↳ 更新（{t['conversation_id']}）：{t['text'][:60]}")
        return
    if args[0] == "--outbox":
        for m in tools._load("outbox.jsonl"):
            print(f"  → {m['to']}  {m['subject']}\n    {m['body'].replace(chr(10), ' / ')[:120]}")
        return

    DB.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(saver)
        if args[0] == "inbox":
            total_calls = 0
            for m in INBOX:
                config = {"configurable": {"thread_id": m["conversation_id"]}}
                if interrupted(graph, config):
                    print(f"[{m['conversation_id']}] {m['id']} 这条线程还停在人工审核上，先不处理")
                    continue
                before = (graph.get_state(config).values or {}).get("model_calls", 0)
                print(f"[{m['conversation_id']}] {m['id']}")
                run(graph, config, {"conversation_id": m["conversation_id"], "incoming": m})
                after = graph.get_state(config).values.get("model_calls", 0)
                total_calls += after - before
            print(f"\n{len(INBOX)} 封邮件，模型调用 {total_calls} 次")
            return
        # 人工审核的决定
        conversation_id, decision = args[0], " ".join(args[1:])
        config = {"configurable": {"thread_id": conversation_id}}
        if not interrupted(graph, config):
            print(f"[{conversation_id}] 没有停在人工审核上")
            return
        print(f"[{conversation_id}] 坐席：{decision}")
        run(graph, config, Command(resume=decision))


if __name__ == "__main__":
    main()
