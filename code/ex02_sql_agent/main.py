"""跑例子 2：
    uv run python -m ex02_sql_agent.main "哪个国家的客户最多？"
    uv run python -m ex02_sql_agent.main --thread q1 "销售额最高的前 5 位艺术家是谁？"
    uv run python -m ex02_sql_agent.main --resume q1 accept
    uv run python -m ex02_sql_agent.main --resume q1 reject
    uv run python -m ex02_sql_agent.main --resume q1 "edit:SELECT ..."
    uv run python -m ex02_sql_agent.main --show-graph

不带 --thread 时 thread_id 取问题的哈希。停在批准那一步的查询，换个进程用同一个
thread_id --resume 接着跑。第一次运行会下载 Chinook.db（约 900KB）到 data/。
"""

import hashlib
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ex02_sql_agent.graph import build_graph

DB = Path(__file__).parent / "data" / "checkpoints.sqlite"


def run(graph, run_input, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    for update in graph.stream(run_input, config=config, stream_mode="updates"):
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            v = info.value
            print(f"[等待批准] thread={thread_id}")
            print(f"  问题：{v['question']}")
            print(f"  SQL：{v['sql']}")
            print(f"  说明：{v['reason']}")
            print(f"  用 --resume {thread_id} accept | reject | \"edit:...\" 继续")
            continue
        for node, changed in update.items():
            for line in changed.get("trace", []):
                print(f"  {line}")
    values = graph.get_state(config).values
    if values.get("answer"):
        print(f"\n回答：{values['answer']}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    DB.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(DB)) as saver:
        graph = build_graph(saver)
        if args[0] == "--show-graph":
            print(graph.get_graph().draw_mermaid())
            return
        if args[0] == "--resume":
            run(graph, Command(resume=" ".join(args[2:])), args[1])
            return
        if args[0] == "--thread":
            thread_id, question = args[1], " ".join(args[2:])
        else:
            question = " ".join(args)
            thread_id = "q-" + hashlib.sha1(question.encode()).hexdigest()[:6]
        print(f"=== {question}  (thread={thread_id}) ===")
        run(graph, {"question": question, "trace": []}, thread_id)


if __name__ == "__main__":
    main()
