"""三个"外部系统"的替身。真实场景里它们是文档库、工单系统、邮件服务的 API；
这里一个查本地 JSON，两个往本地文件追加一行。

这些函数不是给模型选的工具（这张图里模型不选工具），是节点里直接调用的
普通函数——第 1 期讲的第二档：流程是人画的，模型只在几个节点里填空。
"""

import json
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).parent / "data"
DOCS = json.loads((DATA / "docs.json").read_text())
TICKETS = DATA / "tickets.jsonl"
OUTBOX = DATA / "outbox.jsonl"


class TransientError(RuntimeError):
    """模拟文档库偶发超时。设了环境变量 FLAKY_DOCS=1 时，每次调用有一半概率抛它。"""


def search_docs(query: str, top_k: int = 2) -> list[str]:
    """按关键词命中数排序，返回最相关的几条文档。够教学用，真实场景换成 embedding 检索。"""
    if os.environ.get("FLAKY_DOCS") == "1" and random.random() < 0.5:
        print("  [search_docs] 文档库超时（FLAKY_DOCS 注入），抛 TransientError", file=sys.stderr)
        raise TransientError("文档库超时（FLAKY_DOCS 注入）")
    words = [w for w in query.replace("，", " ").replace("？", " ").split() if w]
    scored = []
    for doc in DOCS:
        hits = sum(1 for w in words if w in doc["text"] or w in doc["title"])
        hits += sum(2 for kw in doc["keywords"] if kw in query)
        if hits:
            scored.append((hits, f"{doc['title']}：{doc['text']}"))
    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_k]]


def create_ticket(summary: str, sender: str, urgency: str) -> str:
    ticket_id = f"BUG-{uuid.uuid4().hex[:6].upper()}"
    with TICKETS.open("a") as f:
        f.write(json.dumps({"id": ticket_id, "summary": summary, "sender": sender,
                            "urgency": urgency, "at": datetime.now().isoformat(timespec="seconds")},
                           ensure_ascii=False) + "\n")
    return ticket_id


def send_email(to: str, subject: str, body: str) -> None:
    with OUTBOX.open("a") as f:
        f.write(json.dumps({"to": to, "subject": subject, "body": body,
                            "at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
