"""第 8 期的状态。`messages` 跟第 3-7 期一样。`retrieved` 是"当节点"那个
方案专用的——检索节点把召回的 FAQ 写在这里，`agent` 节点读出来拼进
系统提示词；"当工具"那个方案从头到尾不写这个字段。"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # add_messages 是给对话记录专用的 reducer：按 id 追加或替换，不是简单的 list 相加
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: str
