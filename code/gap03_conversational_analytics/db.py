"""数据库这一侧：造一份可复现的订单库、读 schema、校验、只读执行。没有一行调模型。

例子 2 用的是官方的 Chinook（音乐商店）。这一篇换成这本书自己的旅行订单——业务用户问的是
"上个月哪个目的地卖得最好"，数据得像那回事。数据用固定种子生成，谁跑都是同一份。
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).parent / "data"
DB_PATH = DATA / "bookings.sqlite"
ROW_CAP = 200
TODAY = date(2026, 9, 4)

PRODUCTS = [
    # (id, 名称, 品类, 城市, 国家, 单价 USD)
    ("P01", "东京迪士尼一日票", "景点门票", "东京", "日本", 80),
    ("P02", "大阪周游卡", "交通卡", "大阪", "日本", 44),
    ("P03", "富士山一日游", "一日游", "东京", "日本", 150),
    ("P04", "北海道包车三日", "包车", "札幌", "日本", 600),
    ("P05", "京都和服体验", "体验", "京都", "日本", 35),
    ("P06", "首尔乐天世界门票", "景点门票", "首尔", "韩国", 45),
    ("P07", "釜山一日游", "一日游", "釜山", "韩国", 95),
    ("P08", "曼谷大皇宫导览", "一日游", "曼谷", "泰国", 40),
    ("P09", "曼谷机场接送", "包车", "曼谷", "泰国", 30),
    ("P10", "新加坡环球影城", "景点门票", "新加坡", "新加坡", 75),
    ("P11", "巴厘岛包车一日", "包车", "巴厘岛", "印尼", 55),
    ("P12", "巴厘岛浮潜体验", "体验", "巴厘岛", "印尼", 60),
]
COUNTRIES = ["中国大陆", "香港", "台湾", "新加坡", "马来西亚", "韩国", "美国"]
# 每月基础量 × 季节系数；日本 2026 年逐月走高
SEASON = {1: 0.9, 2: 1.1, 3: 0.9, 4: 1.0, 5: 0.9, 6: 0.8, 7: 1.3, 8: 1.4, 9: 0.9, 10: 1.1, 11: 0.8, 12: 1.2}
REFUND_RATE = {"包车": 0.12, "一日游": 0.07, "景点门票": 0.04, "交通卡": 0.03, "体验": 0.05, "酒店": 0.08}


def _generate(conn: sqlite3.Connection) -> None:
    rng = random.Random(7)
    conn.executescript("""
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY, name TEXT, category TEXT,
            destination_city TEXT, destination_country TEXT, unit_price_usd REAL);
        CREATE TABLE bookings (
            booking_id TEXT PRIMARY KEY, order_date DATE, travel_date DATE, product_id TEXT,
            customer_country TEXT, units INTEGER, amount_usd REAL,
            status TEXT  -- confirmed / refunded / cancelled
        );
    """)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", PRODUCTS)
    rows, n = [], 0
    d = date(2025, 1, 1)
    while d < TODAY:
        base = 4 * SEASON[d.month]
        for pid, _name, cat, _city, country, price in PRODUCTS:
            growth = 1.0
            if country == "日本" and d.year == 2026:
                growth = 1.0 + 0.06 * d.month
            if cat == "包车" and d.year == 2026 and d.month >= 6:
                growth *= 0.7  # 包车 2026 年夏天掉了一截
            k = rng.random() * base * growth
            for _ in range(int(k) + (1 if rng.random() < k - int(k) else 0)):
                n += 1
                units = rng.choice([1, 1, 2, 2, 2, 3, 4])
                travel = d + timedelta(days=rng.randint(3, 60))
                r = rng.random()
                status = "refunded" if r < REFUND_RATE[cat] else ("cancelled" if r < REFUND_RATE[cat] + 0.02 else "confirmed")
                rows.append((f"KL-{100000 + n}", d.isoformat(), travel.isoformat(), pid, rng.choice(COUNTRIES),
                             units, round(units * price * rng.uniform(0.9, 1.05), 2), status))
        d += timedelta(days=1)
    conn.executemany("INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?)", rows)


def ensure_db() -> Path:
    if not DB_PATH.exists():
        DATA.mkdir(exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            _generate(conn)
        print(f"[db] 第一次运行，已生成 {DB_PATH.name}")
    return DB_PATH


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{ensure_db()}?mode=ro", uri=True)  # 只读连接，例子 2 的做法


def load_schema() -> str:
    with connect() as conn:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    return "\n\n".join(r[0] for r in rows)


class QueryRejected(Exception):
    pass


def check_query(sql: str) -> None:
    """例子 2 的两道校验：只放行单条 SELECT/WITH；数据库自己 EXPLAIN。"""
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
        columns = [c[0] for c in cur.description]
        rows = [list(r) for r in cur.fetchmany(ROW_CAP)]
    return columns, rows
