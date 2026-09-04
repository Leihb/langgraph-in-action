"""例子 4 的状态。`results` 带 reducer：几个来源并行跑完各自追加一条，不覆盖。"""

import operator
from typing import Annotated, Literal, TypedDict

Source = Literal["wiki", "tickets", "chat"]


class Classification(TypedDict):
    source: Source
    query: str          # 针对这个来源改写过的子问题


class SourceResult(TypedDict):
    source: Source
    result: str
    seconds: float


class RouterState(TypedDict, total=False):
    question: str
    classifications: list[Classification]
    results: Annotated[list[SourceResult], operator.add]
    final_answer: str


class SourceInput(TypedDict):
    """Send 派给每个来源节点的输入：只有它自己的子问题，看不见别的来源在查什么。"""

    source: Source
    query: str
