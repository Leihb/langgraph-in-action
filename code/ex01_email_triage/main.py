"""跑例子 1：
    uv run python -m ex01_email_triage.main <email_id>            # 处理一封邮件（data/emails.json 里的 id）
    uv run python -m ex01_email_triage.main --resume <email_id> approve
    uv run python -m ex01_email_triage.main --resume <email_id> reject
    uv run python -m ex01_email_triage.main --resume <email_id> "edit:改好的草稿全文"
    uv run python -m ex01_email_triage.main --all                  # 五封全跑一遍，看各走哪条路
    uv run python -m ex01_email_triage.main --show-graph

thread_id 就用 email_id：一封邮件一条线，停在人工审核的邮件，换个进程、
换一天，用同一个 id --resume 就接着走。
"""

import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ex01_email_triage.graph import EMAILS, build_graph

DB = Path(__file__).parent / "data" / "checkpoints.sqlite"


def run(graph, run_input, email_id: str) -> None:
    config = {"configurable": {"thread_id": email_id}}
    for update in graph.stream(run_input, config=config, stream_mode="updates"):
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            v = info.value
            print(f"[等待人工审核] 邮件 {v['email_id']} 来自 {v['from']}｜{v['subject']}")
            print(f"  分类：{v['classification']['intent']} / {v['classification']['urgency']}｜{v['classification']['summary']}")
            print(f"  草稿：{(v['draft'] or '(分类阶段直接转人工，还没有草稿)')[:200]}")
            print(f"  用 --resume {v['email_id']} approve | reject | \"edit:...\" 继续")
            continue
        for node, changed in update.items():
            for line in changed.get("trace", []):
                print(f"  {line}")
    snapshot = graph.get_state(config)
    if snapshot.values.get("sent"):
        print(f"  已回复 {snapshot.values['sender']}：{snapshot.values['draft'][:120]}...")
    elif snapshot.values.get("review") == "reject":
        print("  人工拒绝，不回复。")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(saver)
        if args[0] == "--show-graph":
            print(graph.get_graph().draw_mermaid())
        elif args[0] == "--resume":
            email_id, decision = args[1], " ".join(args[2:])
            run(graph, Command(resume=decision), email_id)
        elif args[0] == "--all":
            for email_id in EMAILS:
                print(f"\n=== {email_id}：{EMAILS[email_id]['subject']} ===")
                run(graph, {"email_id": email_id, "trace": []}, email_id)
        else:
            email_id = args[0]
            print(f"=== {email_id}：{EMAILS[email_id]['subject']} ===")
            run(graph, {"email_id": email_id, "trace": []}, email_id)


if __name__ == "__main__":
    main()
