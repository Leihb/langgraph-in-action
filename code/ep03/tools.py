"""两个工具。docstring 是给模型看的，它靠这段话决定什么时候调、传什么参数。"""

import json
from datetime import date, timedelta
from pathlib import Path

from langchain_core.tools import tool

DATA = Path(__file__).parent / "data"
ORDERS = json.loads((DATA / "orders.json").read_text())
POLICIES = json.loads((DATA / "policies.json").read_text())


@tool
def get_order(order_id: str) -> str:
    """查订单。传订单号（形如 KL-778），返回客人姓名、商品编号、出行日期、数量。查不到返回错误说明。"""
    order = ORDERS.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id}"
    # 出行日期按"今天 + N 天"算，这样示例数据不会过期
    travel_date = date.today() + timedelta(days=order["travel_in_days"])
    return json.dumps(
        {
            "order_id": order_id,
            "customer": order["customer"],
            "product_id": order["product_id"],
            "product_name": POLICIES[order["product_id"]]["name"],
            "travel_date": travel_date.isoformat(),
            "quantity": order["quantity"],
        },
        ensure_ascii=False,
    )


@tool
def get_policy(product_id: str, topic: str) -> str:
    """查商品政策。product_id 是商品编号（形如 SKU-1001），topic 只能是 reschedule（改期）、refund（退款）、usage（使用方式）三者之一。返回政策原文。"""
    product = POLICIES.get(product_id)
    if product is None:
        return f"没有找到商品 {product_id}"
    if topic not in ("reschedule", "refund", "usage"):
        return f"topic 只能是 reschedule / refund / usage，收到的是 {topic}"
    return f"{product['name']} 的 {topic} 政策：{product[topic]}"


TOOLS = [get_order, get_policy]
