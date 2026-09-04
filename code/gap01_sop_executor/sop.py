"""读 SOP 文件。SOP 是 YAML 里的一串步骤，执行器按下标一步步走。"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

SOP_DIR = Path(__file__).parent / "sops"


def load_sops() -> dict[str, dict]:
    sops = {}
    for p in sorted(SOP_DIR.glob("*.yaml")):
        sop = yaml.safe_load(p.read_text(encoding="utf-8"))
        sops[sop["name"]] = sop
    return sops


def step_index(sop: dict, step_id: str) -> int:
    for i, s in enumerate(sop["steps"]):
        if s["id"] == step_id:
            return i
    raise KeyError(step_id)


def when_holds(step: dict, facts: dict) -> bool:
    """`when` 是一个很小的表达式，只能看 facts 里的字段。缺的字段当 None。
    真实系统里应换成白名单的规则引擎，这里为了让 SOP 文件可读，用 eval 演示。"""
    expr = step.get("when")
    if not expr:
        return True
    scope = defaultdict(lambda: None, facts)
    try:
        return bool(eval(expr, {"__builtins__": {}}, scope))  # noqa: S307
    except TypeError:
        return False  # 比如 amount 还没填就拿去比大小


def resolve_args(step: dict, facts: dict) -> dict[str, Any]:
    """args 的值是 facts 里的字段名，换成实际值。"""
    return {k: facts.get(v) for k, v in (step.get("args") or {}).items()}
