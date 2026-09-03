"""skill 注册表：import 时扫一遍 skills/ 目录，把 frontmatter 和正文分开。

跟练习 16 的做法一样，"发现 + 注入"在这一层完成：清单（name + description）
进系统提示词，正文谁都看不见，除非有人调 load_skill 主动去拿。
"""

from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).parent / "skills"


def _parse(path: Path) -> dict:
    text = path.read_text()
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front)
    meta["body"] = body.strip()
    return meta


SKILLS = {p.parent.name: _parse(p) for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))}
SKILL_NAMES = list(SKILLS.keys())


def build_available_skills_prompt() -> str:
    lines = [f"- {name}：{meta['description']}" for name, meta in SKILLS.items()]
    return "\n".join(lines)
