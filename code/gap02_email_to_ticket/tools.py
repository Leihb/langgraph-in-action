"""工单库和发信，都是假的：工单落 data/tickets.jsonl，邮件落 data/outbox.jsonl。"""

import json
from pathlib import Path

DATA = Path(__file__).parent / "data"


def _load(name: str) -> list[dict]:
    p = DATA / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(name: str, record: dict) -> None:
    DATA.mkdir(exist_ok=True)
    with (DATA / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_open_ticket(order_id: str | None, category: str) -> dict | None:
    """同一订单、同一类别、还没解决的工单——有就归并，不新建。"""
    if not order_id:
        return None
    for t in _load("tickets.jsonl"):
        if t["_op"] == "create" and t["order_id"] == order_id and t["category"] == category and t["status"] == "open":
            return t
    return None


def create_ticket(ticket: dict) -> str:
    n = sum(1 for t in _load("tickets.jsonl") if t.get("_op") == "create") + 1
    no = f"T-{n:04d}"
    _append("tickets.jsonl", {"_op": "create", "ticket_no": no, "status": "open", **ticket})
    return no


def append_update(ticket_no: str, conversation_id: str, text: str) -> None:
    _append("tickets.jsonl", {"_op": "update", "ticket_no": ticket_no, "conversation_id": conversation_id, "text": text})


def send_email(to: str, subject: str, body: str) -> None:
    _append("outbox.jsonl", {"to": to, "subject": subject, "body": body})


def all_tickets() -> list[dict]:
    return _load("tickets.jsonl")
