"""第 10 期新增三个字段，专门为并行核对订单这条支线服务。

`order_ids`、`check_aspect`、`order_id` 都是"写一次、读一次"的临时值，没有
reducer——每个 step 里只有一个节点会写它们，用不上合并逻辑。

`order_reports` 不一样：`lookup_order` 会被 `Send` 同时派发好几份，几份并行的
调用会在同一个 step 里各自写一次这个字段，LangGraph 要求这种"同一 key 被并发
写"的情况必须有 reducer 来说明怎么合并，否则直接报错。用 `operator.or_`（字典
合并）而不是 `operator.add`（列表追加）：每个 `lookup_order` 只知道自己查的
那一个订单号，写 `{订单号: 结果}` 这样的单键字典，多份并行结果按键合并成一个
大字典，互不覆盖。用字典还有一个好处——旧一轮查询留下的键不会干扰下一轮：
`aggregate` 只按当次的 `order_ids` 去字典里取值，取不到的键（不管是不是这一轮的）
自然被忽略。"""

import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_skills: Annotated[list[str], operator.add]
    order_ids: NotRequired[list[str]]
    order_id: NotRequired[str]
    check_aspect: NotRequired[str]
    order_reports: Annotated[dict[str, str], operator.or_]
