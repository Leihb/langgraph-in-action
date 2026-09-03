"""第 9 期的状态。`messages` 跟第 3-8 期一样。`loaded_skills` 是这一期
新加的：记这场对话已经加载过哪些 skill 正文，`operator.add` 让它像日志
一样往后追加，不是覆盖——但会不会真的追加，由 `load_skill` 工具自己
判断，见 `tools.py`。"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_skills: Annotated[list[str], operator.add]
