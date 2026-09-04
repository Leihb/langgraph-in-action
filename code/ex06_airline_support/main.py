"""跑例子 6：
    uv run python -m ex06_airline_support.main --version 4 --reset --script          # 重置库，跑一遍脚本对话（遇到闸门自动批准）
    uv run python -m ex06_airline_support.main --version 3 t1 "你好，我的航班是几点？"
    uv run python -m ex06_airline_support.main --version 3 --resume t1 y             # 批准
    uv run python -m ex06_airline_support.main --version 3 --resume t1 "先别订，我再想想"   # 拒绝并说明原因
    uv run python -m ex06_airline_support.main --version 4 --state t1

乘客固定是官方教程里那位（passenger_id 3442 587242，两张 BSL↔CDG 的票）。第一次运行下载 114MB 数据库。
"""

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ex06_airline_support import db
from ex06_airline_support.graphs import BUILDERS

PASSENGER = "3442 587242"
CHECKPOINTS = Path(__file__).parent / "data" / "checkpoints.sqlite"

# 官方 14 句英文对话，压成 10 句中文——覆盖查航班、查政策、改签、订酒店、租车、订活动
SCRIPT = [
    "你好，我的航班是几点？",
    "我能把航班改到今天更早一点吗？",
    "那改到下周吧，最近的一班就行",
    "住宿和交通呢？我在巴塞尔要住 7 天",
    "订一家价格适中的酒店就行，你推荐的那家",
    "租车有什么选择？最便宜的那个订 7 天",
    "巴塞尔有什么景点推荐？我喜欢博物馆",
    "挑一个订上",
]


def run_turn(graph, run_input, config: dict, auto_approve: bool) -> None:
    """跑一轮；碰到闸门就打印待批的调用，auto_approve 时自动放行，否则退出等 --resume。"""
    while True:
        interrupted = None
        for update in graph.stream(run_input, config=config, stream_mode="updates"):
            if "__interrupt__" in update:
                (info,) = update["__interrupt__"]
                interrupted = info.value
                continue
            for node, changed in update.items():
                if not isinstance(changed, dict):
                    continue
                if changed.get("dialog_state"):
                    print(f"  ⇢ dialog_state {changed['dialog_state']!r}（节点 {node}）")
                for msg in changed.get("messages", []):
                    if getattr(msg, "tool_calls", None):
                        for c in msg.tool_calls:
                            print(f"  [{node}] 调用 {c['name']}({_short(c['args'])})")
                    elif msg.type == "tool":
                        print(f"  [{node}] 工具返回：{_short(str(msg.content), 100)}")
                    elif msg.type == "ai" and msg.content:
                        print(f"  [{node}] 助理：{_short(msg.content, 300)}")
        if interrupted is None:
            return
        print(f"  [闸门] 待批准：{[(p['tool'], _short(p['args'])) for p in interrupted['pending']]}")
        if not auto_approve:
            print("  用 --resume <thread> y 批准，或 --resume <thread> \"原因\" 拒绝")
            return
        print("  [闸门] 自动批准 y")
        run_input = Command(resume="y")


def _short(x, n: int = 80) -> str:
    s = str(x).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument("--reset", action="store_true", help="从备份重置数据库并平移航班时间")
    p.add_argument("--script", action="store_true", help="跑内置的脚本对话，闸门自动批准")
    p.add_argument("--turns", type=int, default=len(SCRIPT), help="脚本只跑前 N 轮")
    p.add_argument("--resume", nargs=2, metavar=("THREAD", "DECISION"))
    p.add_argument("--state", metavar="THREAD")
    p.add_argument("rest", nargs="*", help="<thread> <message>")
    args = p.parse_args()

    if args.reset:
        db.reset()
    CHECKPOINTS.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINTS)) as saver:
        graph = BUILDERS[args.version](saver)
        cfg = lambda t: {"configurable": {"thread_id": f"v{args.version}-{t}", "passenger_id": PASSENGER}}  # noqa: E731

        if args.state:
            v = graph.get_state(cfg(args.state)).values
            print(f"dialog_state={v.get('dialog_state')}  messages={len(v.get('messages', []))}")
            return
        if args.resume:
            thread, decision = args.resume
            run_turn(graph, Command(resume=decision), cfg(thread), auto_approve=False)
            return
        if args.script:
            for i, q in enumerate(SCRIPT[: args.turns], 1):
                print(f"\n=== v{args.version} 第 {i} 轮  客人：{q}")
                run_turn(graph, {"messages": [HumanMessage(q)]}, cfg("script"), auto_approve=True)
            return
        if len(args.rest) < 2:
            p.error("需要 <thread> <message>")
        thread, text = args.rest[0], " ".join(args.rest[1:])
        print(f"=== v{args.version} [{thread}] 客人：{text}")
        run_turn(graph, {"messages": [HumanMessage(text)]}, cfg(thread), auto_approve=False)


if __name__ == "__main__":
    main()
