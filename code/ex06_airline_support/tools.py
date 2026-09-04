"""官方教程的全部工具移植过来，docstring 改成中文。分四组：航班、租车、酒店、景点，
外加一个查政策的。每组里"查"是安全工具，"订/改/取消"是敏感工具——这个分法第三版起用。

passenger_id 从 runtime.config 里拿：调图的人在 config 里传，模型不知道也改不了，
一个乘客看不到另一个乘客的票。
"""

import sqlite3
from datetime import date, datetime, timedelta, timezone

from langchain.tools import ToolRuntime, tool

from ex06_airline_support import db


def _rows(cur) -> list[dict]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _passenger(runtime: ToolRuntime) -> str:
    pid = runtime.config.get("configurable", {}).get("passenger_id")
    if not pid:
        raise ValueError("config 里没有 passenger_id")
    return pid


def user_flights(passenger_id: str) -> list[dict]:
    """给图的第一个节点直接调用的普通函数（第二版起先查好乘客信息再进助理）。"""
    with db.connect() as conn:
        return _rows(conn.execute("""
            SELECT t.ticket_no, t.book_ref, f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport,
                   f.scheduled_departure, f.scheduled_arrival, bp.seat_no, tf.fare_conditions
            FROM tickets t JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
                 JOIN flights f ON tf.flight_id = f.flight_id
                 JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
            WHERE t.passenger_id = ?""", (passenger_id,)))


# ---------- 政策 ----------

@tool
def lookup_policy(query: str) -> str:
    """查公司政策（改签、退票、行李、支付等规则）。做任何改动之前先查一下允不允许。"""
    words = [w for w in query.lower().replace("?", " ").split() if len(w) >= 3]
    scored = sorted(db.faq_sections(), key=lambda s: -sum(s.lower().count(w) for w in words))
    return "\n\n".join(scored[:2])


# ---------- 航班 ----------

@tool
def fetch_user_flight_information(runtime: ToolRuntime) -> list[dict]:
    """查当前乘客的全部机票、对应航班和座位。"""
    return user_flights(_passenger(runtime))


@tool
def search_flights(departure_airport: str | None = None, arrival_airport: str | None = None,
                   start_time: str | None = None, end_time: str | None = None, limit: int = 20) -> list[dict]:
    """按出发机场、到达机场（三字码）和出发时间范围（ISO 日期或日期时间）搜航班。"""
    q, params = "SELECT * FROM flights WHERE 1=1", []
    if departure_airport:
        q += " AND departure_airport = ?"; params.append(departure_airport)
    if arrival_airport:
        q += " AND arrival_airport = ?"; params.append(arrival_airport)
    if start_time:
        q += " AND scheduled_departure >= ?"; params.append(start_time)
    if end_time:
        q += " AND scheduled_departure <= ?"; params.append(end_time)
    q += " ORDER BY scheduled_departure LIMIT ?"; params.append(limit)
    with db.connect() as conn:
        return _rows(conn.execute(q, params))


@tool
def update_ticket_to_new_flight(ticket_no: str, new_flight_id: int, runtime: ToolRuntime) -> str:
    """把乘客的机票改到另一个航班。起飞前不足 3 小时的航班不允许改。"""
    pid = _passenger(runtime)
    with db.connect() as conn:
        new = conn.execute("SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
                           (new_flight_id,)).fetchone()
        if not new:
            return "没有这个航班 id"
        dep = datetime.fromisoformat(new[2])
        if (dep - datetime.now(dep.tzinfo)) < timedelta(hours=3):
            return f"不允许改到起飞前不足 3 小时的航班（该航班 {dep} 起飞）"
        if not conn.execute("SELECT 1 FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)).fetchone():
            return "没有这张机票"
        if not conn.execute("SELECT 1 FROM tickets WHERE ticket_no = ? AND passenger_id = ?", (ticket_no, pid)).fetchone():
            return f"当前乘客 {pid} 不是机票 {ticket_no} 的持有人"
        conn.execute("UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?", (new_flight_id, ticket_no))
        conn.commit()
    return "机票已改到新航班"


@tool
def cancel_ticket(ticket_no: str, runtime: ToolRuntime) -> str:
    """取消乘客的机票。"""
    pid = _passenger(runtime)
    with db.connect() as conn:
        if not conn.execute("SELECT 1 FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)).fetchone():
            return "没有这张机票"
        if not conn.execute("SELECT 1 FROM tickets WHERE ticket_no = ? AND passenger_id = ?", (ticket_no, pid)).fetchone():
            return f"当前乘客 {pid} 不是机票 {ticket_no} 的持有人"
        conn.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
        conn.commit()
    return "机票已取消"


# ---------- 租车 / 酒店 / 景点：三组长得一样 ----------

def _search(table: str, location: str | None, name: str | None, extra_sql: str = "", extra_params: list | None = None) -> list[dict]:
    q, params = f"SELECT * FROM {table} WHERE 1=1", []
    if location:
        q += " AND location LIKE ?"; params.append(f"%{location}%")
    if name:
        q += " AND name LIKE ?"; params.append(f"%{name}%")
    q += extra_sql; params += extra_params or []
    with db.connect() as conn:
        return _rows(conn.execute(q, params))


def _set(table: str, row_id: int, assignments: dict, label: str) -> str:
    if not assignments:
        return f"{label} {row_id}：没有要改的内容"
    with db.connect() as conn:
        cur = conn.execute(f"UPDATE {table} SET {', '.join(k + ' = ?' for k in assignments)} WHERE id = ?",
                           (*assignments.values(), row_id))
        conn.commit()
        return f"{label} {row_id} 已更新" if cur.rowcount else f"没有 id 为 {row_id} 的{label}"


@tool
def search_car_rentals(location: str | None = None, name: str | None = None, price_tier: str | None = None,
                       start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    """按地点、公司名搜租车。教学数据很少，价格档和日期不做过滤。"""
    return _search("car_rentals", location, name)


@tool
def book_car_rental(rental_id: int) -> str:
    """按 id 预订一辆租车。"""
    return _set("car_rentals", rental_id, {"booked": 1}, "租车")


@tool
def update_car_rental(rental_id: int, start_date: str | None = None, end_date: str | None = None) -> str:
    """改租车的起止日期。"""
    return _set("car_rentals", rental_id, {k: v for k, v in [("start_date", start_date), ("end_date", end_date)] if v}, "租车")


@tool
def cancel_car_rental(rental_id: int) -> str:
    """取消一辆租车。"""
    return _set("car_rentals", rental_id, {"booked": 0}, "租车")


@tool
def search_hotels(location: str | None = None, name: str | None = None, price_tier: str | None = None,
                  checkin_date: str | None = None, checkout_date: str | None = None) -> list[dict]:
    """按地点、酒店名搜酒店。price_tier 可选：Midscale / Upper Midscale / Upscale / Luxury。教学数据很少，价格档和日期不做过滤。"""
    return _search("hotels", location, name)


@tool
def book_hotel(hotel_id: int) -> str:
    """按 id 预订酒店。"""
    return _set("hotels", hotel_id, {"booked": 1}, "酒店")


@tool
def update_hotel(hotel_id: int, checkin_date: str | None = None, checkout_date: str | None = None) -> str:
    """改酒店的入住/退房日期。"""
    return _set("hotels", hotel_id, {k: v for k, v in [("checkin_date", checkin_date), ("checkout_date", checkout_date)] if v}, "酒店")


@tool
def cancel_hotel(hotel_id: int) -> str:
    """取消酒店预订。"""
    return _set("hotels", hotel_id, {"booked": 0}, "酒店")


@tool
def search_trip_recommendations(location: str | None = None, name: str | None = None, keywords: str | None = None) -> list[dict]:
    """按地点、名字、关键词（逗号分开）搜景点和活动推荐。"""
    extra, params = "", []
    if keywords:
        kws = [k.strip() for k in keywords.split(",") if k.strip()]
        extra = " AND (" + " OR ".join("keywords LIKE ?" for _ in kws) + ")"
        params = [f"%{k}%" for k in kws]
    return _search("trip_recommendations", location, name, extra, params)


@tool
def book_excursion(recommendation_id: int) -> str:
    """按 id 预订一个景点/活动。"""
    return _set("trip_recommendations", recommendation_id, {"booked": 1}, "景点")


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """更新一个景点/活动预订的备注。"""
    return _set("trip_recommendations", recommendation_id, {"details": details}, "景点")


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """取消一个景点/活动预订。"""
    return _set("trip_recommendations", recommendation_id, {"booked": 0}, "景点")


FLIGHT_SAFE, FLIGHT_SENSITIVE = [search_flights], [update_ticket_to_new_flight, cancel_ticket]
CAR_SAFE, CAR_SENSITIVE = [search_car_rentals], [book_car_rental, update_car_rental, cancel_car_rental]
HOTEL_SAFE, HOTEL_SENSITIVE = [search_hotels], [book_hotel, update_hotel, cancel_hotel]
TRIP_SAFE, TRIP_SENSITIVE = [search_trip_recommendations], [book_excursion, update_excursion, cancel_excursion]

SAFE_TOOLS = [fetch_user_flight_information, lookup_policy, *FLIGHT_SAFE, *CAR_SAFE, *HOTEL_SAFE, *TRIP_SAFE]
SENSITIVE_TOOLS = [*FLIGHT_SENSITIVE, *CAR_SENSITIVE, *HOTEL_SENSITIVE, *TRIP_SENSITIVE]
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS
