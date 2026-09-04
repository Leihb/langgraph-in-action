"""例子 3 的状态：在 create_agent 自带的 AgentState（messages 等）上加几个字段。

`current_step` 是整张图的方向盘。它由工具改（工具返回 Command），由中间件读
（按它换提示词和工具集）。模型自己碰不到这个字段。
"""

from typing import Literal, NotRequired

from langchain.agents.middleware import AgentState

SupportStep = Literal["identify", "classify", "resolve"]


class SupportState(AgentState):
    current_step: NotRequired[SupportStep]
    # identify 阶段由 lookup_order 写入
    order_id: NotRequired[str]
    customer: NotRequired[str]
    product_id: NotRequired[str]
    product_name: NotRequired[str]
    travel_date: NotRequired[str]
    # classify 阶段由 record_issue 写入
    issue_type: NotRequired[Literal["reschedule", "refund", "other"]]
    # resolve 阶段由 provide_solution 写入
    solution: NotRequired[str]
