"""三个工具。前两个跟第 3-4 期一样，cancel_order 是这一期新加的：
执行前会停下来等人工审批，docstring 告诉模型有这么一步。"""

import json
from datetime import date, timedelta
from pathlib import Path

from langchain_core.tools import tool
from langgraph.types import interrupt

DATA = Path(__file__).parent / "data"
ORDERS = json.loads((DATA / "orders.json").read_text())
POLICIES = json.loads((DATA / "policies.json").read_text())
CANCEL_LOG = DATA / "cancel_log.txt"


@tool
def get_order(order_id: str) -> str:
    """查订单。传订单号（形如 KL-778），返回客人姓名、商品编号、出行日期、数量。查不到返回错误说明。"""
    order = ORDERS.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id}"
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


@tool
def cancel_order(order_id: str) -> str:
    """取消订单。传订单号。这个操作会先停下来等人工审批，人工同意才真正取消，
    拒绝的话订单不受影响。"""
    order = ORDERS.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id}"

    # 这一行写在 interrupt 之前，纯粹是为了这一期让你亲眼看见"恢复时会重跑"——
    # 真实业务不会把这种记账代码放在 interrupt 前面。
    with CANCEL_LOG.open("a") as f:
        f.write(f"{order_id}\n")

    decision = interrupt(
        {
            "action": "cancel_order",
            "order_id": order_id,
            "customer": order["customer"],
            "product": POLICIES[order["product_id"]]["name"],
        }
    )
    if decision != "approve":
        return f"人工拒绝，订单 {order_id} 未取消"

    # 真正的取消动作（调订单服务、扣库存之类）必须放在这一行之后：
    # interrupt 之前的代码会在恢复时重跑，这一行不会。
    return f"订单 {order_id} 已取消"


TOOLS = [get_order, get_policy, cancel_order]
