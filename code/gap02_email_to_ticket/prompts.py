"""模型只干一件事：把一段邮件往来抽成工单字段。"""

EXTRACT = """你是客服后台的建单助手。下面是一条邮件线程里的全部邮件（按时间顺序），把它抽成工单字段。

{thread}

只返回 JSON，键固定为：
{{"ticket_type": "customer_demand"（客人自己的诉求）| "feedback"（表扬/建议/评价，不需要处理订单）| "merchant_request"（商户/供应商发来的）| "not_a_request"（广告、通知、跟客服无关）,
  "category": "refund" | "amendment"（改日期/改人数等）| "cancellation" | "compensation"（要赔偿/补偿）| "invoice" | "inquiry" | "praise" | "other",
  "order_id": 订单号，写成 KL-三位数字，邮件里没有就 null,
  "customer_name": 客人署名或 null,
  "request": 一句话说清客人要什么,
  "target_date": 改期目标日 YYYY-MM-DD（用收信日期推算相对日期），没有就 null,
  "amount": 客人提到的金额数字，没有就 null,
  "reason": 客人给的原因原话，没有就 null,
  "language": 邮件语言，如 "zh" / "en"}}
规则：一封邮件只建一个类别，多个诉求取主要的那个（要钱的优先于要发票的）。
邮件里没写的字段填 null，不要猜。发件人是谁、是不是下单人，不用你判断，代码会查。{feedback}"""

FEEDBACK = """

上一次抽取有这些问题，请修正后重新返回完整 JSON：
{problems}"""
