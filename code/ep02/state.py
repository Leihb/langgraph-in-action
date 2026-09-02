"""第 2 期的状态：一张图里所有节点共享的那份数据。"""

import operator
from typing import Annotated, TypedDict


class DraftState(TypedDict):
    # 输入
    product_id: str
    question: str
    # 中间结果：每个字段由某一个节点写入，后写的覆盖先写的
    category: str
    policy: str
    draft: str
    # 带 reducer 的字段：每个节点往里追加一条，不覆盖
    trace: Annotated[list[str], operator.add]
