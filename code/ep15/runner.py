"""第 15 期：给第 14 期那个客服 agent 跑行为评测。

    uv run python -m ep15.runner sync-dataset        # 把 cases.json 同步成 Langfuse 的 Dataset
    uv run python -m ep15.runner run                 # 实跑：每条用例一个全新 agent，录制 + 打分 + 对比基线
    uv run python -m ep15.runner run --local         # 不走 Langfuse Dataset，直接用本地 cases.json
    uv run python -m ep15.runner replay runs/<ts>.json   # 回放：不调 agent，对旧录制重新打分
    uv run python -m ep15.runner run --update-baseline   # 把这次的失败集写成新基线

被测对象是 `ep14` 的图和工具——这一期没有自己的 agent 代码，只有用例、
打分器、裁判和这个运行器。MCP 工具（第 7 期的 get_current_time）没接进来：
没有一条用例测它，少一个子进程也让每条用例的启动快一些。

三层分工：
- 运行器（Langfuse 的 `run_experiment`）：并发、单条失败隔离、每条用例一条
  trace、结果挂成一个 Dataset Run。
- 代码打分器（`graders.py`）：确定性断言，看工具调用和末态。
- 裁判（`judge.py`）：只管 rubric。

Langfuse 不管的那部分自己做：task 函数里怎么起一个干净的 agent、怎么注入
前置状态；录制落盘让打分器改了能回放；失败集按"用例:打分器"为键跟基线比，
只有新失败才算失败。
"""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from common import settings
from ep15.graph import build_graph
from ep15.tools import TOOLS
from ep15.graders import grade
from ep15.judge import judge, rubric_fingerprint

HERE = Path(__file__).parent
CASES = json.loads((HERE / "cases.json").read_text())
RUNS = HERE / "runs"
BASELINE = HERE / "baseline.json"
DATASET_NAME = "customer-agent-evals"  # 名字里别带斜杠，Langfuse 的 get_dataset 会把它当成路径


# ---------- 1. 跑一条用例：全新 agent、注入前置状态、录下发生了什么 ----------

async def run_case(case: dict) -> dict:
    saver, store = InMemorySaver(), InMemoryStore()
    user_id = case["user_id"]
    if case["state"].get("memory"):
        await store.aput((user_id, "memory"), "note", {"text": case["state"]["memory"]})

    graph = build_graph(saver, store, TOOLS)
    config = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid.uuid4().hex[:6]}", "user_id": user_id}}

    tool_calls, answers, interrupt_payload = [], [], None
    for turn in case["turns"]:
        async for update in graph.astream({"messages": [HumanMessage(turn)]}, config=config, stream_mode="updates"):
            if "__interrupt__" in update:
                (info,) = update["__interrupt__"]
                interrupt_payload = info.value
                continue
            for node, changed in update.items():
                if node != "agent":
                    continue
                parts = changed if isinstance(changed, list) else [changed]
                for part in parts:
                    for msg in part.get("messages", []):
                        if msg.tool_calls:
                            tool_calls += [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]
                        elif msg.content:
                            answers.append(msg.content)
        if interrupt_payload is not None:
            break  # 停在 interrupt 就不再喂下一轮，这是被测行为的一部分

    snapshot = await graph.aget_state(config)
    note = await store.aget((user_id, "memory"), "note")
    return {
        "turns": case["turns"],
        "tool_calls": tool_calls,
        "final_reply": answers[-1] if answers else None,
        "interrupted": bool(snapshot.interrupts),
        "interrupt_payload": interrupt_payload,
        "loaded_skills": list(snapshot.values.get("loaded_skills", [])),
        "memory_after": note.value["text"] if note else None,
    }


# ---------- 2. 接到 Langfuse 的 run_experiment 上 ----------

CASE_BY_ID = {c["id"]: c for c in CASES}


def _case_of(item) -> dict:
    # metadata 里只放 id，不放整条用例：Langfuse 会把 metadata 逐字段传播成
    # trace 属性，超过 200 字符的值（比如 rubric）会被丢掉并打一条警告。
    meta = item.metadata if hasattr(item, "metadata") else item["metadata"]
    return CASE_BY_ID[meta["case_id"]]


async def task(*, item, **_) -> dict:
    return await run_case(_case_of(item))


def code_graders(*, output, expected_output, **_):
    from langfuse import Evaluation

    return [
        Evaluation(name=key, value=ok, comment=msg)
        for key, (ok, msg) in grade(output, expected_output).items()
    ]


def rubric_judge(*, output, expected_output, **_):
    from langfuse import Evaluation

    rubric = expected_output.get("rubric")
    if not rubric:
        return []
    verdict = judge(output, rubric)
    if verdict["verdict"] == "JUDGE_ERROR":
        return [Evaluation(name="judge_error", value=True, comment=verdict["reason"])]
    return [Evaluation(name="rubric", value=verdict["verdict"] == "PASS",
                       comment=f"{verdict['reason']} [{verdict['fingerprint']}]")]


def pass_rate(*, item_results, **_):
    from langfuse import Evaluation

    flags = [e.value for r in item_results for e in r.evaluations if isinstance(e.value, bool) and e.name != "judge_error"]
    return Evaluation(name="pass_rate", value=round(sum(flags) / len(flags), 3) if flags else 0.0,
                      comment=f"{sum(flags)}/{len(flags)} 个断言通过")


def local_items(cases: list[dict]) -> list[dict]:
    return [{"input": {"turns": c["turns"], "state": c["state"]}, "expected_output": c["expected"],
             "metadata": {"case_id": c["id"]}} for c in cases]


def sync_dataset(cases: list[dict]) -> None:
    from langfuse import get_client

    lf = get_client()
    lf.create_dataset(name=DATASET_NAME, description="第 15 期：客服 agent 行为评测用例（来自 code/ep15/cases.json）")
    for c in cases:
        lf.create_dataset_item(dataset_name=DATASET_NAME, id=c["id"],
                               input={"turns": c["turns"], "state": c["state"]},
                               expected_output=c["expected"], metadata={"case_id": c["id"]})
    print(f"已同步 {len(cases)} 条用例到 Langfuse dataset {DATASET_NAME!r}")


# ---------- 3. 录制、汇总、跟基线比 ----------

def summarize(entries: list[dict]) -> set[str]:
    """打印每条用例的结果，返回失败集 {"用例id:打分器", ...}。"""
    failures = set()
    for e in entries:
        marks = []
        for key, r in e["graders"].items():
            marks.append(f"{'✓' if r['pass'] else '✗'} {key}")
            if not r["pass"]:
                failures.add(f"{e['id']}:{key}")
        if e.get("judge"):
            v = e["judge"]["verdict"]
            marks.append(f"{'✓' if v == 'PASS' else '✗' if v == 'FAIL' else '?'} rubric")
            if v == "FAIL":
                failures.add(f"{e['id']}:rubric")
        print(f"{e['id']:<36} {'  '.join(marks)}")
        for key, r in e["graders"].items():
            if not r["pass"]:
                print(f"{'':<36}   {key}: {r['msg']}")
        if e.get("judge") and e["judge"]["verdict"] != "PASS":
            print(f"{'':<36}   rubric: {e['judge']['verdict']} — {e['judge']['reason']}")
    return failures


def compare_baseline(failures: set[str], update: bool) -> int:
    known = set(json.loads(BASELINE.read_text())["known_failures"]) if BASELINE.exists() else set()
    new, fixed = failures - known, known - failures
    print(f"\n失败 {len(failures)} 项；基线里已知 {len(known)} 项；新失败 {len(new)}，已修好 {len(fixed)}")
    for f in sorted(new):
        print(f"  NEW   {f}")
    for f in sorted(fixed):
        print(f"  FIXED {f}")
    if update:
        BASELINE.write_text(json.dumps({"known_failures": sorted(failures)}, ensure_ascii=False, indent=2) + "\n")
        print(f"基线已更新：{BASELINE}")
        return 0
    return 1 if new else 0


def cmd_run(args) -> int:
    from langfuse import get_client

    cases = [c for c in CASES if not args.only or c["id"] in args.only]
    lf = get_client()
    # get_dataset() 返回的是 DatasetClient，run_experiment 要的是它的 .items 列表
    data = local_items(cases) if args.local else lf.get_dataset(DATASET_NAME).items
    if not args.local and args.only:
        data = [i for i in data if i.metadata["case_id"] in args.only]
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = lf.run_experiment(
        name="customer-agent-evals", run_name=run_name,
        description=f"model={settings.MODEL_NAME}",
        data=data, task=task,
        evaluators=[code_graders, rubric_judge], run_evaluators=[pass_rate],
        max_concurrency=args.concurrency,
    )
    lf.flush()

    entries = []
    for r in result.item_results:
        case = _case_of(r.item)
        graders = {e.name: {"pass": bool(e.value), "msg": e.comment} for e in r.evaluations
                   if e.name not in ("rubric", "judge_error")}
        judge_ev = next((e for e in r.evaluations if e.name in ("rubric", "judge_error")), None)
        judge_out = None
        if judge_ev is not None:
            judge_out = ({"verdict": "JUDGE_ERROR", "reason": judge_ev.comment} if judge_ev.name == "judge_error"
                         else {"verdict": "PASS" if judge_ev.value else "FAIL", "reason": judge_ev.comment})
            judge_out["fingerprint"] = rubric_fingerprint(case["expected"]["rubric"]) if case["expected"].get("rubric") else None
        entries.append({"id": case["id"], "expected": case["expected"], "record": r.output,
                        "graders": graders, "judge": judge_out, "trace_id": r.trace_id})

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{run_name}.json"
    out.write_text(json.dumps({"run_name": run_name, "model": settings.MODEL_NAME,
                               "items": entries}, ensure_ascii=False, indent=2) + "\n")
    print(f"\n录制已写到 {out}\n")
    failures = summarize(entries)
    for ev in result.run_evaluations:
        print(f"\n{ev.name} = {ev.value}  ({ev.comment})")
    return compare_baseline(failures, args.update_baseline)


def cmd_replay(args) -> int:
    """不调 agent、不调裁判（除非 --rejudge），只对录制重跑代码打分器。"""
    data = json.loads(Path(args.recording).read_text())
    for e in data["items"]:
        # expected 以当前 cases.json 为准——回放的意义就是“断言改了，对旧录制重新判”；
        # 用例已经被删掉的才退回录制里存的那份。
        expected = CASE_BY_ID.get(e["id"], {}).get("expected", e["expected"])
        e["expected"] = expected
        e["graders"] = {k: {"pass": ok, "msg": msg} for k, (ok, msg) in grade(e["record"], expected).items()}
        rubric = expected.get("rubric")
        if args.rejudge and rubric:
            v = judge(e["record"], rubric)
            e["judge"] = {"verdict": v["verdict"], "reason": v["reason"], "fingerprint": v["fingerprint"]}
        elif rubric and e.get("judge") and e["judge"].get("fingerprint") != rubric_fingerprint(rubric):
            # rubric 或裁判模型变了，录制里的判定已经不能用——标出来，不算通过也不算失败
            e["judge"] = {**e["judge"], "verdict": "STALE", "reason": "rubric 或裁判模型已变，用 --rejudge 重判"}
    if args.rejudge:
        Path(args.recording).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"回放 {args.recording}（model={data['model']}）\n")
    return compare_baseline(summarize(data["items"]), args.update_baseline)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--local", action="store_true", help="不走 Langfuse Dataset，直接用 cases.json")
    r.add_argument("--only", nargs="*", help="只跑这些用例 id")
    r.add_argument("--concurrency", type=int, default=3)
    r.add_argument("--update-baseline", action="store_true")
    rp = sub.add_parser("replay")
    rp.add_argument("recording")
    rp.add_argument("--rejudge", action="store_true")
    rp.add_argument("--update-baseline", action="store_true")
    sub.add_parser("sync-dataset")
    args = p.parse_args()

    if args.cmd == "sync-dataset":
        sync_dataset(CASES)
        return
    sys.exit(cmd_run(args) if args.cmd == "run" else cmd_replay(args))


if __name__ == "__main__":
    main()
