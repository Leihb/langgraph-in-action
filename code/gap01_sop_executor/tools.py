"""SOP 每一步调的内部接口，全是假数据。

注意：这些函数不给模型看、不是 LangChain 工具。调哪个、传什么参数，写在 SOP 文件里，
由执行器按步骤调用。模型在这一期没有"选工具"的权力。
"""

import json
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"

ORDERS = {
    "KL-778": {"order_id": "KL-778", "customer": "王女士", "product": "东京迪士尼一日票 x2", "amount": 320,
               "currency": "USD", "payment_status": "paid", "fraud_status": "pass", "guest_checkout": False,
               "compensations": []},
    "KL-901": {"order_id": "KL-901", "customer": "陈先生", "product": "大阪周游卡", "amount": 88,
               "currency": "USD", "payment_status": "unpaid", "fraud_status": "pass", "guest_checkout": False,
               "compensations": []},
    "KL-315": {"order_id": "KL-315", "customer": "李先生", "product": "富士山一日游", "amount": 150,
               "currency": "USD", "payment_status": "paid", "fraud_status": "pass", "guest_checkout": False,
               "compensations": [{"compensation_no": "RA2609010001", "amount": 30, "status": "pending_approval"}]},
    "KL-502": {"order_id": "KL-502", "customer": "赵女士", "product": "北海道包车三日", "amount": 1800,
               "currency": "USD", "payment_status": "paid", "fraud_status": "pass", "guest_checkout": False,
               "compensations": []},
}
# 支付网关还能原路退多少（订单纯现金支付 − 已退 − 已补偿）
GATEWAY_BALANCE = {"KL-778": 60, "KL-315": 120, "KL-502": 1800}


def get_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        raise LookupError(f"没有订单 {order_id}")
    return order


def get_gateway_balance(order_id: str) -> int:
    return GATEWAY_BALANCE.get(order_id, 0)


def apply_compensation(order_id: str, amount: float, comp_type: str, reason: str,
                       bank_account: str | None, approver: str) -> dict:
    DATA.mkdir(exist_ok=True)
    no = f"RA{time.strftime('%y%m%d%H%M%S')}"
    auto_part = amount if comp_type == "credit" else min(amount, get_gateway_balance(order_id))
    manual_part = 0 if comp_type == "credit" else max(0, amount - auto_part)
    record = {"compensation_no": no, "order_id": order_id, "amount": amount, "comp_type": comp_type,
              "reason": reason, "approver": approver, "status": "pending_compensation",
              "resources": [{"kind": "gateway" if comp_type == "cash" else "credit", "amount": auto_part}]
              + ([{"kind": "manual_transfer", "amount": manual_part, "account": bank_account}] if manual_part else [])}
    with (DATA / "compensations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def add_booking_note(order_id: str, compensation: dict) -> str:
    DATA.mkdir(exist_ok=True)
    text = (f"补偿单 {compensation['compensation_no']}：{compensation['comp_type']} {compensation['amount']} USD，"
            f"原因：{compensation['reason']}，审批：{compensation['approver']}")
    with (DATA / "notes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"order_id": order_id, "note": text}, ensure_ascii=False) + "\n")
    return text


TOOLS = {
    "get_order": get_order,
    "get_gateway_balance": get_gateway_balance,
    "apply_compensation": apply_compensation,
    "add_booking_note": add_booking_note,
}
