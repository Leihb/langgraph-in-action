"""check 步骤跑的规则。每条规则：拿 facts，返回 (过不过, 不过的原因)。

规则是代码，理由跟例子 1 的路由一样：这些判断错了会出事（给没付钱的订单打钱），
不该让模型"参考政策自己判断"。SOP 文件里只写规则名。
"""


def order_paid(facts: dict) -> tuple[bool, str]:
    o = facts["order"]
    return o["payment_status"] == "paid", f"订单 {o['order_id']} 未支付，不能发起补偿"


def not_in_fraud_review(facts: dict) -> tuple[bool, str]:
    o = facts["order"]
    return o["fraud_status"] == "pass", f"订单 {o['order_id']} 风控状态 {o['fraud_status']}，不能发起补偿"


def no_open_compensation(facts: dict) -> tuple[bool, str]:
    o = facts["order"]
    open_ones = [c for c in o["compensations"] if c["status"] == "pending_approval"]
    if open_ones:
        return False, f"订单 {o['order_id']} 已有待审批的补偿单 {open_ones[0]['compensation_no']}（{open_ones[0]['amount']} USD），先处理它"
    return True, ""


RULES = {
    "order_paid": order_paid,
    "not_in_fraud_review": not_in_fraud_review,
    "no_open_compensation": no_open_compensation,
}
