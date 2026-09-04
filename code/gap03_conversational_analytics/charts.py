"""选图和画图，全是代码。

选图规则看结果的"形状"：几列、哪几列是数字、第一列像不像日期、多少行。模型可以提一个偏好
（用户说"换成折线"），但偏好跟形状不合就忽略、按规则来，并在返回里说明。

图表规格（spec）是给前端用的结构化 JSON；终端里用字符画一份，让人不用前端也看得见结果。
"""

import re
from collections import OrderedDict

DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
VALID_TYPES = ("number", "bar", "line", "pie", "table")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_cols(columns, rows) -> list[int]:
    return [i for i in range(len(columns)) if rows and all(_is_num(r[i]) or r[i] is None for r in rows)]


def _looks_like_date(rows, i) -> bool:
    return all(isinstance(r[i], str) and DATE_RE.match(r[i]) for r in rows)


def pivot_long(columns, rows):
    """三列长表（x, 系列, 值）转宽表（x, 系列1, 系列2, ...）。模型按 GROUP BY month, category 写出来
    的就是长表，多序列图要的是宽表，这一步是代码的活。"""
    xs = OrderedDict()
    series = OrderedDict()
    for x, s, v in rows:
        series.setdefault(s, None)
        xs.setdefault(x, {})[s] = v
    cols = [columns[0]] + [str(s) for s in series]
    out = [[x] + [vals.get(s, 0) for s in series] for x, vals in xs.items()]
    return cols, out


def choose(columns, rows, hint: str | None) -> tuple[str, list[str], list[list], str]:
    """返回 (图类型, 列, 行, 一句说明)。行/列可能被 pivot 过。"""
    if not rows:
        return "table", columns, rows, "没有数据"
    nums = _numeric_cols(columns, rows)
    cats = [i for i in range(len(columns)) if i not in nums]
    note = ""
    # 三列长表 → 宽表
    if len(columns) == 3 and len(cats) == 2 and len(nums) == 1 and nums[0] == 2:
        columns, rows = pivot_long(columns, rows)
        nums = list(range(1, len(columns)))
        cats = [0]
        note = "已把长表转成宽表；"
    if len(rows) == 1 and len(nums) == 1 and len(columns) == 1:
        kind = "number"
    elif len(cats) == 1 and cats[0] == 0 and 1 <= len(nums) <= 5 and len(rows) <= 40:
        kind = "line" if _looks_like_date(rows, 0) else "bar"
    else:
        kind = "table"
    if hint and hint in VALID_TYPES and hint != kind:
        ok = (hint == "pie" and kind == "bar" and len(nums) == 1 and len(rows) <= 8) \
             or (hint in ("bar", "line") and kind in ("bar", "line")) \
             or hint == "table"
        if ok:
            kind = hint
        else:
            note += f"用户想要 {hint}，但结果形状不适合，按规则用 {kind}；"
    return kind, columns, rows, note.rstrip("；")


def partial_period_caveat(columns, rows, today) -> str:
    """时间序列的最后一个点如果是"还没过完"的当月/当年，标出来。
    这一篇真机第一次跑就撞上：'过去 12 个月'带进了只过了 3 天的 9 月，模型的两句话全在讲
    '9 月断崖式下滑'。这种误读不该靠模型自己识破，代码知道今天是几号。"""
    if not rows or not _looks_like_date(rows, 0):
        return ""
    last = str(rows[-1][0])
    if last == today.strftime("%Y-%m"):
        return f"{last} 是当前月，只到 {today.day} 号，不是完整月份"
    if last == today.strftime("%Y"):
        return f"{last} 是当前年，只到 {today.month} 月，不是完整年份"
    return ""


def spec(kind, columns, rows, title) -> dict:
    """给前端的规格。字段名有意跟 Vega-Lite 靠近，但不承诺兼容。"""
    if kind == "number":
        return {"type": "number", "title": title, "value": rows[0][0], "label": columns[0]}
    if kind == "table":
        return {"type": "table", "title": title, "columns": columns, "rows": rows}
    return {"type": kind, "title": title, "x": columns[0], "series": columns[1:],
            "data": [dict(zip(columns, r)) for r in rows]}


# ---------- 终端渲染 ----------

def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
    return str(v)


def render(kind, columns, rows, title, width=40) -> str:
    lines = [f"▍{title}"]
    if kind == "number":
        return "\n".join(lines + [f"   {_fmt(rows[0][0])}  ({columns[0]})"])
    if kind == "table":
        w = [max(len(str(c)), *(len(_fmt(r[i])) for r in rows)) for i, c in enumerate(columns)]
        lines.append("   " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(columns)))
        for r in rows[:30]:
            lines.append("   " + "  ".join(_fmt(v).rjust(w[i]) if _is_num(v) else _fmt(v).ljust(w[i]) for i, v in enumerate(r)))
        if len(rows) > 30:
            lines.append(f"   … 共 {len(rows)} 行")
        return "\n".join(lines)
    if kind == "pie":
        total = sum(r[1] or 0 for r in rows) or 1
        lw = max(len(str(r[0])) for r in rows)
        for r in rows:
            share = (r[1] or 0) / total
            lines.append(f"   {str(r[0]).ljust(lw)} {'●' * max(1, round(share * width))} {share:.0%}")
        return "\n".join(lines)
    glyphs = "█▓▒░▚"
    series = columns[1:]
    vmax = max((abs(v or 0) for r in rows for v in r[1:]), default=1) or 1
    lw = max(len(str(r[0])) for r in rows)
    if kind == "bar":
        for r in rows:
            for si, v in enumerate(r[1:]):
                label = str(r[0]).ljust(lw) if si == 0 else " " * lw
                lines.append(f"   {label} {glyphs[si % len(glyphs)] * max(1, round(abs(v or 0) / vmax * width))} {_fmt(v)}")
        if len(series) > 1:
            lines.append("   " + "  ".join(f"{glyphs[i % len(glyphs)]} {s}" for i, s in enumerate(series)))
        return "\n".join(lines)
    # line：纵向 10 行，每列一个 x
    h = 10
    grid = [[" "] * len(rows) for _ in range(h)]
    marks = "●◆■▲◇"
    for si in range(len(series)):
        for xi, r in enumerate(rows):
            v = r[1 + si] or 0
            y = h - 1 - round(v / vmax * (h - 1))
            grid[y][xi] = marks[si % len(marks)] if grid[y][xi] == " " else "✱"
    for y, row in enumerate(grid):
        left = _fmt(vmax * (h - 1 - y) / (h - 1)).rjust(8) if y in (0, h // 2, h - 1) else " " * 8
        lines.append(f"   {left} ┤ " + "  ".join(row))
    xs = [str(r[0])[-5:] if len(str(r[0])) > 5 else str(r[0]) for r in rows]
    lines.append("   " + " " * 8 + " └ " + "  ".join(x[-2:].rjust(1) for x in xs))
    lines.append("   " + " " * 11 + f"{xs[0]} … {xs[-1]}")
    if len(series) > 1:
        lines.append("   " + "  ".join(f"{marks[i % len(marks)]} {s}" for i, s in enumerate(series)))
    return "\n".join(lines)
