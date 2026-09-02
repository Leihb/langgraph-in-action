CATEGORIES = ("reschedule", "refund", "usage")

CLASSIFY = """你是客服系统里的分类器。判断客人的问题属于哪一类，只输出类别英文单词，不要输出别的：
- reschedule：改期、换日期、换时间
- refund：退款、取消、退钱
- usage：怎么用、怎么入园、怎么兑换、注意事项

客人的问题：{question}"""

DRAFT = """你是客服，正在给客人起草回复。只根据下面的政策原文回答，不要编造政策里没有的条款。
语气礼貌简洁，中文，不超过 80 字。

商品：{name}
政策原文：{policy}
客人的问题：{question}"""
