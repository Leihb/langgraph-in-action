"""工单的字段，和"抽出来的字段对不对"的三类判断。

模型抽的字段（TicketDraft）和代码填的字段（priority / sla_hours / customer_email……）分开。
校验结果分三类，后面三条路：
  FORMAT    模型抽错了格式（订单号少个横线、日期不是 YYYY-MM-DD、类别不在枚举里）→ 把错误喂回去让模型重抽
  MISSING   客人没给（没写订单号、改期没说改到哪天）→ 回信问客人，这条线等回复
  MISMATCH  跟系统对不上（订单不存在、发件人不是下单人、改期日期已过）→ 转人工
"""

import re
from datetime import date, datetime
from typing import Literal, TypedDict

from gap02_email_to_ticket.orders import MERCHANT_DOMAINS, get_order

TICKET_TYPES = ("customer_demand", "feedback", "merchant_request", "not_a_request")
CATEGORIES = ("refund", "amendment", "cancellation", "compensation", "invoice", "inquiry", "praise", "other")
ORDER_RE = re.compile(r"^KL-\d{3}$")
# 各类别在建单前必须有的、且只有客人能给的字段
REQUIRED_FROM_CUSTOMER = {"amendment": ["target_date"], "refund": ["reason"], "compensation": ["reason"]}
# 出行日期距收信不到 48 小时的退改类诉求算紧急
URGENT_CATEGORIES = {"refund", "amendment", "cancellation"}
SLA_HOURS = {"urgent": 12, "high": 24, "normal": 48, "low": 72}


class TicketDraft(TypedDict, total=False):
    ticket_type: str
    category: str
    order_id: str | None
    customer_name: str | None
    request: str
    target_date: str | None
    amount: float | None
    reason: str | None
    language: str


class Problem(TypedDict):
    kind: Literal["FORMAT", "MISSING", "MISMATCH"]
    field: str
    message: str


def validate(draft: TicketDraft, sender: str, received_at: str) -> list[Problem]:
    problems: list[Problem] = []
    tt, cat = draft.get("ticket_type"), draft.get("category")
    if tt not in TICKET_TYPES:
        problems.append({"kind": "FORMAT", "field": "ticket_type", "message": f"ticket_type 必须是 {TICKET_TYPES} 之一，收到 {tt!r}"})
    if cat not in CATEGORIES:
        problems.append({"kind": "FORMAT", "field": "category", "message": f"category 必须是 {CATEGORIES} 之一，收到 {cat!r}"})
    if tt in ("not_a_request", "feedback") or problems:
        return problems  # 不是诉求 / 表扬不需要订单；枚举都错了先修枚举

    oid = draft.get("order_id")
    if oid is None:
        problems.append({"kind": "MISSING", "field": "order_id", "message": "邮件里没有订单号"})
    elif not ORDER_RE.match(str(oid)):
        problems.append({"kind": "FORMAT", "field": "order_id", "message": f"order_id 必须写成 KL-三位数字（如 KL-778），收到 {oid!r}"})
    else:
        order = get_order(oid)
        if order is None:
            problems.append({"kind": "MISMATCH", "field": "order_id", "message": f"系统里没有订单 {oid}"})
        elif tt == "customer_demand" and order["email"].lower() != sender.lower():
            problems.append({"kind": "MISMATCH", "field": "sender", "message": f"发件人 {sender} 不是订单 {oid} 的下单人（{order['email']}）"})

    td = draft.get("target_date")
    if td is not None:
        try:
            d = date.fromisoformat(str(td))
            if d < datetime.fromisoformat(received_at).date():
                problems.append({"kind": "MISMATCH", "field": "target_date", "message": f"改期目标日 {td} 早于收信日期"})
        except ValueError:
            problems.append({"kind": "FORMAT", "field": "target_date", "message": f"target_date 必须是 YYYY-MM-DD 或 null，收到 {td!r}（收信日期 {received_at[:10]}）"})

    amt = draft.get("amount")
    if amt is not None and not isinstance(amt, (int, float)):
        problems.append({"kind": "FORMAT", "field": "amount", "message": f"amount 必须是数字或 null，收到 {amt!r}"})

    for f in REQUIRED_FROM_CUSTOMER.get(cat, []):
        if draft.get(f) in (None, ""):
            problems.append({"kind": "MISSING", "field": f, "message": f"{cat} 类工单必须有 {f}，邮件里没有"})
    return problems


def priority_for(draft: TicketDraft, received_at: str) -> str:
    """优先级是代码算的，不是模型评的。"""
    if draft.get("ticket_type") == "merchant_request":
        return "high"
    if draft.get("ticket_type") in ("feedback", "not_a_request"):
        return "low"
    order = get_order(draft.get("order_id") or "")
    if order and draft.get("category") in URGENT_CATEGORIES:
        hours = (datetime.fromisoformat(order["travel_date"]) - datetime.fromisoformat(received_at)).total_seconds() / 3600
        if 0 <= hours < 48:  # 出行日已过的不算紧急，那是另一类问题
            return "urgent"
    return "high" if draft.get("category") in ("compensation", "cancellation") else "normal"


def is_merchant(sender: str) -> bool:
    return sender.split("@")[-1].lower() in MERCHANT_DOMAINS
