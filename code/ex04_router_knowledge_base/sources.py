"""三个知识来源，各自一个搜索工具、一个小 agent。

真实场景里这三个工具是 Confluence/飞书文档、工单系统、IM 的搜索 API；这里是三份
JSON 上的关键词匹配。每个来源一个 `create_agent`：拿到针对它的子问题，自己决定
搜什么词、搜几次，最后用一段话汇报"在我这里找到了什么"。三个 agent 互相看不见。
"""

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool

from common.llm import chat_model

DATA = Path(__file__).parent / "data"
WIKI = json.loads((DATA / "wiki.json").read_text())
TICKETS = json.loads((DATA / "tickets.json").read_text())
CHAT = json.loads((DATA / "chat.json").read_text())


def _hits(query: str, records: list[dict], fields: tuple[str, ...], top_k: int = 3) -> list[dict]:
    words = [w for w in query.replace("，", " ").replace("、", " ").split() if len(w) >= 2]
    scored = []
    for r in records:
        text = " ".join(str(r[f]) for f in fields)
        score = sum(text.count(w) for w in words) + sum(2 for w in words if len(w) >= 4 and w in text)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


@tool
def search_wiki(query: str) -> str:
    """搜内部政策与流程文档。传几个关键词（空格分开），返回最相关的几篇文档原文。"""
    hits = _hits(query, WIKI, ("title", "text"))
    return "\n\n".join(f"《{h['title']}》\n{h['text']}" for h in hits) or "没有匹配的文档"


@tool
def search_tickets(query: str) -> str:
    """搜历史工单（客人遇到过的问题和当时怎么处理的）。传几个关键词，返回最相关的几条。"""
    hits = _hits(query, TICKETS, ("title", "resolution"))
    return "\n\n".join(f"[{h['id']} {h['date']}] {h['title']}\n处理：{h['resolution']}" for h in hits) or "没有匹配的工单"


@tool
def search_chat(query: str) -> str:
    """搜客服群的聊天记录（同事之间的经验、提醒、口头约定）。传几个关键词，返回最相关的几条。"""
    hits = _hits(query, CHAT, ("text",))
    return "\n\n".join(f"[{h['date']} {h['author']}] {h['text']}" for h in hits) or "没有匹配的聊天记录"


def _agent(tool_fn, name: str):
    return create_agent(
        chat_model(),
        tools=[tool_fn],
        system_prompt=(
            f"你负责{name}这一个来源。用搜索工具找跟问题相关的内容，可以换关键词多搜一两次。"
            "最后用一段中文汇报：找到了什么、出处是哪条（文档标题 / 工单号 / 谁在哪天说的）。"
            "没找到就直说没找到，不要编。"
        ),
    )


AGENTS = {
    "wiki": _agent(search_wiki, "内部政策与流程文档"),
    "tickets": _agent(search_tickets, "历史工单"),
    "chat": _agent(search_chat, "客服群聊天记录"),
}

SOURCE_DESCRIPTIONS = {
    "wiki": "内部政策与流程文档：正式的规则、流程、规范",
    "tickets": "历史工单：客人遇到过的具体问题和当时的处理结果",
    "chat": "客服群聊天记录：同事之间的经验、提醒、口头约定、谁说过什么",
}
