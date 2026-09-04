"""例子 1 的状态。

一条规矩贯穿整张图：**state 里存原始数据，不存拼好的文本**。分类结果存成
结构化字段，检索结果存成列表，草稿存成一段纯文本——每个节点要什么格式，
自己在节点里拼。这样换一个节点想换种拼法不用回头改别人，调试时看 state
也一眼能看出每个字段是谁写的、值对不对。
"""

import operator
from typing import Annotated, Literal, TypedDict


class EmailClassification(TypedDict):
    """分类节点让模型按这个结构输出，四个字段都是路由要用的。"""

    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailState(TypedDict, total=False):
    # 输入：读邮件节点从 data/emails.json 里取出来写进去
    email_id: str
    sender: str
    subject: str
    body: str
    # 中间结果：一个字段只由一个节点写，后写覆盖先写
    classification: EmailClassification
    search_results: list[str]
    ticket_id: str
    draft: str
    review: str  # 人工审核的决定：approve / edit / reject
    sent: bool
    # 每个节点追加一条，看整条路怎么走的
    trace: Annotated[list[str], operator.add]
