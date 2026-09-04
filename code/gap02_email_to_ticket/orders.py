"""订单系统的假数据。跟这本书其他章节同一批订单，多了下单人邮箱和出行日期。"""

ORDERS = {
    "KL-778": {"order_id": "KL-778", "customer": "王慧", "email": "wang.hui@example.com",
               "product": "东京迪士尼一日票 x2", "travel_date": "2026-09-08", "amount": 320},
    "KL-901": {"order_id": "KL-901", "customer": "陈俊", "email": "chen.jun@example.com",
               "product": "大阪周游卡", "travel_date": "2026-09-06", "amount": 88},
    "KL-315": {"order_id": "KL-315", "customer": "李明", "email": "li.ming@example.com",
               "product": "富士山一日游", "travel_date": "2026-09-05", "amount": 150},
    "KL-502": {"order_id": "KL-502", "customer": "赵敏", "email": "zhao.min@example.com",
               "product": "北海道包车三日", "travel_date": "2026-09-02", "amount": 1800},
}

# 商户（供应商）的邮箱域名：来自这些域名的邮件是商户诉求，不是客人诉求
MERCHANT_DOMAINS = {"hokkaido-charter.co.jp"}


def get_order(order_id: str) -> dict | None:
    return ORDERS.get(order_id)
