"""第 3 期的状态：只有一份对话记录。"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages 是给对话记录专用的 reducer：按 id 追加或替换，不是简单的 list 相加
    messages: Annotated[list[AnyMessage], add_messages]
