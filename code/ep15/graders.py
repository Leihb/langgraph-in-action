"""代码打分器：`expected` 里除了 `rubric` 以外的每一个键，对应这里一个函数。

输入是一条录制（`runner.py` 跑完一条用例写下的字典），输出 (通过?, 说明)。
全部是确定性判断，不调模型、不花钱，改了打分逻辑可以对着旧录制重放。

录制里用到的字段：
    tool_calls      主 agent 层按顺序发出的工具调用 [{"name", "args"}, ...]
    final_reply     最后一条回答文本（停在 interrupt 时是 None）
    interrupted     这条用例结束时是否停在 interrupt
    loaded_skills   末态 state 里的 loaded_skills
    memory_after    末态 store 里这个用户的笔记文本（没有则 None）
"""

from collections.abc import Callable

Grader = Callable[[dict, object], tuple[bool, str]]


def _names(rec: dict) -> list[str]:
    return [c["name"] for c in rec["tool_calls"]]


def calls_tool(rec: dict, want: list[str]) -> tuple[bool, str]:
    missing = [t for t in want if t not in _names(rec)]
    return (not missing, f"缺少调用 {missing}" if missing else "都调了")


def calls_one_of(rec: dict, options: list[str]) -> tuple[bool, str]:
    hit = [t for t in options if t in _names(rec)]
    return (bool(hit), f"调了 {hit}" if hit else f"{options} 一个都没调")


def never_calls(rec: dict, banned: list[str]) -> tuple[bool, str]:
    bad = [t for t in _names(rec) if t in banned]
    return (not bad, f"不该调却调了 {bad}" if bad else "没碰禁用工具")


def first_tool(rec: dict, want: str) -> tuple[bool, str]:
    names = _names(rec)
    got = names[0] if names else None
    return (got == want, f"第一个工具是 {got}，期望 {want}")


def first_tool_not(rec: dict, banned: str) -> tuple[bool, str]:
    names = _names(rec)
    got = names[0] if names else None
    return (got != banned, f"第一个工具是 {got}")


def max_tool_calls(rec: dict, limit: int) -> tuple[bool, str]:
    n = len(rec["tool_calls"])
    return (n <= limit, f"调了 {n} 次，上限 {limit}")


def tool_args_include(rec: dict, spec: dict) -> tuple[bool, str]:
    """spec 形如 {"get_policy": {"topic": "refund"}}：某个工具至少有一次调用
    带着这些参数值。"""
    for name, want_args in spec.items():
        calls = [c["args"] for c in rec["tool_calls"] if c["name"] == name]
        if not any(all(a.get(k) == v for k, v in want_args.items()) for a in calls):
            return (False, f"{name} 没有一次调用带 {want_args}，实际 {calls}")
    return (True, "参数对上了")


def interrupts(rec: dict, want: bool) -> tuple[bool, str]:
    return (rec["interrupted"] == want, f"interrupted={rec['interrupted']}，期望 {want}")


def reply_includes(rec: dict, needles: list[str]) -> tuple[bool, str]:
    text = rec["final_reply"] or ""
    missing = [s for s in needles if s not in text]
    return (not missing, f"回复里没有 {missing}" if missing else "都有")


def reply_omits(rec: dict, needles: list[str]) -> tuple[bool, str]:
    text = rec["final_reply"] or ""
    bad = [s for s in needles if s in text]
    return (not bad, f"回复里出现了 {bad}" if bad else "都没出现")


def skill_loaded(rec: dict, name: str) -> tuple[bool, str]:
    return (name in rec["loaded_skills"], f"loaded_skills={rec['loaded_skills']}")


def skill_not_loaded(rec: dict, name: str) -> tuple[bool, str]:
    return (name not in rec["loaded_skills"], f"loaded_skills={rec['loaded_skills']}")


def no_skill_load(rec: dict, _: bool) -> tuple[bool, str]:
    return (not rec["loaded_skills"], f"loaded_skills={rec['loaded_skills']}")


def memory_contains(rec: dict, needles: list[str]) -> tuple[bool, str]:
    text = rec["memory_after"] or ""
    missing = [s for s in needles if s not in text]
    return (not missing, f"笔记里没有 {missing}，笔记={text!r}" if missing else "笔记里有")


def memory_not_contains(rec: dict, needles: list[str]) -> tuple[bool, str]:
    text = rec["memory_after"] or ""
    bad = [s for s in needles if s in text]
    return (not bad, f"笔记里出现了 {bad}" if bad else "笔记里没有")


GRADERS: dict[str, Grader] = {
    "calls_tool": calls_tool,
    "calls_one_of": calls_one_of,
    "never_calls": never_calls,
    "first_tool": first_tool,
    "first_tool_not": first_tool_not,
    "max_tool_calls": max_tool_calls,
    "tool_args_include": tool_args_include,
    "interrupts": interrupts,
    "reply_includes": reply_includes,
    "reply_omits": reply_omits,
    "skill_loaded": skill_loaded,
    "skill_not_loaded": skill_not_loaded,
    "no_skill_load": no_skill_load,
    "memory_contains": memory_contains,
    "memory_not_contains": memory_not_contains,
}


def grade(rec: dict, expected: dict) -> dict[str, tuple[bool, str]]:
    """跑 expected 里每一个代码打分器（跳过 rubric，那是裁判的活）。"""
    results = {}
    for key, want in expected.items():
        if key == "rubric":
            continue
        if key not in GRADERS:
            raise KeyError(f"expected 里有不认识的键 {key!r}，可用：{sorted(GRADERS)}")
        results[key] = GRADERS[key](rec, want)
    return results
