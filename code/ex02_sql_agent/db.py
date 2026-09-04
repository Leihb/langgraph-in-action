"""数据库这一侧的全部代码：下载、读 schema、校验、只读执行。没有一行调模型。"""

import sqlite3
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DB_PATH = DATA / "Chinook.db"
# 官方教程用的同一份 Chinook 示例库（一家数字音乐商店：艺术家、专辑、曲目、客户、发票）
DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
ROW_CAP = 50


def ensure_db() -> Path:
    if not DB_PATH.exists():
        DATA.mkdir(exist_ok=True)
        print(f"[db] 第一次运行，下载 Chinook.db（约 900KB）…")
        urllib.request.urlretrieve(DB_URL, DB_PATH)
    return DB_PATH


def connect() -> sqlite3.Connection:
    # mode=ro：连接本身只读。模型生成的 SQL 就算漏过了校验，也写不进去——
    # 这条保证在数据库层，跟提示词、跟校验代码都无关。
    return sqlite3.connect(f"file:{ensure_db()}?mode=ro", uri=True)


def load_schema() -> str:
    """全库 11 张表的建表语句，一共两千来个字符，直接整份给模型，省掉"让模型先选表"那一步。"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return "\n\n".join(r[0] for r in rows)


class QueryRejected(Exception):
    pass


def check_query(sql: str) -> None:
    """两道校验，都是确定性的：
    1. 只放行 SELECT / WITH 开头的单条语句——这是业务规则，写在代码里。
    2. 让数据库自己做 EXPLAIN QUERY PLAN：语法错、表名列名不存在，数据库会报出来，
       比让另一个模型"复查一遍"便宜、准确。
    没过就抛 QueryRejected，原因回给模型改。"""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise QueryRejected("只允许一条语句，不能用分号拼接")
    if not s.upper().startswith(("SELECT", "WITH")):
        raise QueryRejected("只允许 SELECT 查询，这条语句以 %s 开头" % s.split()[0].upper())
    with connect() as conn:
        try:
            conn.execute(f"EXPLAIN QUERY PLAN {s}")
        except sqlite3.Error as e:
            raise QueryRejected(f"数据库校验没过：{e}") from e


def run_query(sql: str) -> tuple[list[str], list[list]]:
    s = sql.strip().rstrip(";")
    with connect() as conn:
        cur = conn.execute(s)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchmany(ROW_CAP)]
    return columns, rows
