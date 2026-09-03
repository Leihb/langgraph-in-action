"""LLM 裁判：只处理 `expected.rubric`，代码判不了的语义判断交给它。

几条硬规则，每条都对应一个真实的失败模式：
- 裁判模型固定、temperature=0——换裁判或改 rubric 都会让历史判定失效，
  所以录制里带着两者的指纹（模型名 + rubric 文本的哈希）。
- 裁判拿到的是完整对话记录（客人的话、每次工具调用和参数、最后的回答），
  它手里没有夹具数据，rubric 必须把决定结果的事实写进去。
- 裁判输出解析不出 PASS/FAIL 算"裁判失败"，跟 agent 失败分开记，不混进通过率。
"""

import hashlib
import json

from langchain_core.messages import HumanMessage, SystemMessage

from common import settings
from common.llm import chat_model

JUDGE_SYSTEM = """You are a strict grader for a customer-service agent's behaviour.
You will be given a rubric and a transcript. Decide PASS or FAIL according to the rubric only.
Do not grade tone, length or politeness unless the rubric asks for it.
Reply with a single JSON object: {"verdict": "PASS" | "FAIL", "reason": "<one sentence>"}"""


def rubric_fingerprint(rubric: str) -> str:
    return f"{settings.MODEL_NAME}:{hashlib.sha1(rubric.encode()).hexdigest()[:10]}"


def _transcript(rec: dict) -> str:
    lines = []
    for t in rec["turns"]:
        lines.append(f"[customer] {t}")
    for c in rec["tool_calls"]:
        lines.append(f"[tool_call] {c['name']}({json.dumps(c['args'], ensure_ascii=False)})")
    if rec["interrupted"]:
        lines.append(f"[interrupt] waiting for human approval: {json.dumps(rec['interrupt_payload'], ensure_ascii=False)}")
    lines.append(f"[agent final reply] {rec['final_reply'] or '(none, stopped at interrupt)'}")
    return "\n".join(lines)


def judge(rec: dict, rubric: str) -> dict:
    """返回 {"verdict": PASS|FAIL|JUDGE_ERROR, "reason": ..., "fingerprint": ...}。"""
    model = chat_model(temperature=0)
    prompt = f"RUBRIC:\n{rubric}\n\nTRANSCRIPT:\n{_transcript(rec)}"
    raw = model.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(prompt)]).content
    fp = rubric_fingerprint(rubric)
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        verdict = str(data.get("verdict", "")).upper()
        if verdict not in ("PASS", "FAIL"):
            raise ValueError(f"verdict={verdict!r}")
        return {"verdict": verdict, "reason": data.get("reason", ""), "fingerprint": fp}
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        return {"verdict": "JUDGE_ERROR", "reason": f"{type(e).__name__}: {e}; raw={raw[:200]!r}", "fingerprint": fp}
