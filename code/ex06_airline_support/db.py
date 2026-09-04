"""数据：官方教程用的 travel2.sqlite（一家航空公司的航班、机票、酒店、租车、景点，114MB）
和 swiss_faq.md（政策问答）。第一次运行自动下载到 data/，之后离线。

官方笔记本用 pandas 把航班日期整体平移到"现在"，让"我的航班几点"这类问题有意义。
这里用纯 sqlite 做同一件事，只平移 flights 表的四个时间列（工具只读这四列）。
`reset()` 从备份重来，每一版图跑之前都重置，保证四版看到的是同一份数据。
"""

import re
import shutil
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).parent / "data"
DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
FAQ_URL = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
BACKUP = DATA / "travel2.backup.sqlite"
DB = DATA / "travel2.sqlite"
FAQ = DATA / "swiss_faq.md"


def ensure_files() -> None:
    DATA.mkdir(exist_ok=True)
    if not BACKUP.exists():
        print("[db] 第一次运行，下载 travel2.sqlite（114MB）…")
        urllib.request.urlretrieve(DB_URL, BACKUP)
    if not FAQ.exists():
        urllib.request.urlretrieve(FAQ_URL, FAQ)
    if not DB.exists():
        reset()


def reset() -> None:
    """从备份复制一份干净的库，再把航班时间平移到现在。"""
    ensure_files() if not BACKUP.exists() else None
    shutil.copy(BACKUP, DB)
    conn = sqlite3.connect(DB)
    (latest,) = conn.execute(
        "SELECT MAX(actual_departure) FROM flights WHERE actual_departure IS NOT NULL AND actual_departure != '\\N'"
    ).fetchone()
    example_time = datetime.fromisoformat(latest)
    diff = datetime.now(example_time.tzinfo) - example_time
    cols = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
    rows = conn.execute(f"SELECT flight_id, {', '.join(cols)} FROM flights").fetchall()

    def shift(v):
        if v is None or v == "\\N":
            return v
        return (datetime.fromisoformat(v) + diff).isoformat(sep=" ")

    conn.executemany(
        f"UPDATE flights SET {', '.join(c + ' = ?' for c in cols)} WHERE flight_id = ?",
        [(*[shift(v) for v in r[1:]], r[0]) for r in rows],
    )
    conn.commit()
    conn.close()
    print(f"[db] 已重置并把航班时间平移 {diff.days} 天")


def connect() -> sqlite3.Connection:
    ensure_files()
    return sqlite3.connect(DB)


def faq_sections() -> list[str]:
    ensure_files()
    return [s.strip() for s in re.split(r"(?=\n##)", FAQ.read_text()) if s.strip()]
