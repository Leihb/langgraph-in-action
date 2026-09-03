"""前六个工具跟第 6-9 期原样一致，一行没改。`check_orders` 是这一期新加的——
它自己不查订单，只是把订单号列表和"要核对的角度"写进状态，交给图的路由去决定
怎么并行处理，见 `graph.py` 里的 `route_after_tools` 和 `lookup_order`。"""

import json
from datetime import date, timedelta
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, interrupt

from ep12 import retrieval
from ep12.registry import SKILLS, SKILL_NAMES

DATA = Path(__file__).parent / "data"
ORDERS = json.loads((DATA / "orders.json").read_text())
POLICIES = json.loads((DATA / "policies.json").read_text())
CANCEL_LOG = DATA / "cancel_log.txt"


@tool
def get_order(order_id: str) -> str:
    """查订单。传订单号（形如 KL-778），返回客人姓名、商品编号、出行日期、数量。查不到返回错误说明。"""
    order = ORDERS.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id}"
    travel_date = date.today() + timedelta(days=order["travel_in_days"])
    return json.dumps(
        {
            "order_id": order_id,
            "customer": order["customer"],
            "product_id": order["product_id"],
            "product_name": POLICIES[order["product_id"]]["name"],
            "travel_date": travel_date.isoformat(),
            "quantity": order["quantity"],
        },
        ensure_ascii=False,
    )


@tool
def get_policy(product_id: str, topic: str) -> str:
    """查商品政策。product_id 是商品编号（形如 SKU-1001），topic 只能是 reschedule（改期）、refund（退款）、usage（使用方式）三者之一。返回政策原文。"""
    product = POLICIES.get(product_id)
    if product is None:
        return f"没有找到商品 {product_id}"
    if topic not in ("reschedule", "refund", "usage"):
        return f"topic 只能是 reschedule / refund / usage，收到的是 {topic}"
    return f"{product['name']} 的 {topic} 政策：{product[topic]}"


@tool
def cancel_order(order_id: str) -> str:
    """取消订单。传订单号。这个操作会先停下来等人工审批，人工同意才真正取消，
    拒绝的话订单不受影响。"""
    order = ORDERS.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id}"

    with CANCEL_LOG.open("a") as f:
        f.write(f"{order_id}\n")

    decision = interrupt(
        {
            "action": "cancel_order",
            "order_id": order_id,
            "customer": order["customer"],
            "product": POLICIES[order["product_id"]]["name"],
        }
    )
    if decision != "approve":
        return f"人工拒绝，订单 {order_id} 未取消"

    return f"订单 {order_id} 已取消"


@tool
def remember_note(note: str, runtime: ToolRuntime) -> str:
    """覆盖跨会话笔记。传笔记的完整内容——这次传的文本会整个替换掉旧笔记，
    需要保留的部分要自己带上，只传新增的一小段会把之前该留的内容丢掉。
    传空字符串等于清空笔记。"""
    user_id = runtime.config["configurable"]["user_id"]
    runtime.store.put((user_id, "memory"), "note", {"text": note})
    return "笔记已更新" if note else "笔记已清空"


@tool
def search_faq(query: str) -> str:
    """查通用政策问答：行李规定、支付方式、发票、儿童票、极端天气、团体优惠、
    电子票、改手机号这类问题。get_policy 只覆盖改期/退款/使用方式三类，
    问不到的都用这个。"""
    hits = retrieval.search_faq(query, top_k=2)
    return "\n".join(f"{h['question']}：{h['answer']}" for h in hits)


@tool(description=(
    "加载一份 skill 正文，拿到某个场景的详细处理办法。可选：" + ", ".join(SKILL_NAMES) + "。"
    "清单里的一句话描述不够用、需要具体步骤和措辞的时候才调这个，不要每次都加载。"
))
def load_skill(name: str, runtime: ToolRuntime) -> Command | str:
    if name not in SKILLS:
        return f"没有这个 skill：{name}，可选：{', '.join(SKILL_NAMES)}"

    loaded = runtime.state.get("loaded_skills", [])
    if name in loaded:
        msg = f"{name} 这一场对话已经加载过了，不重复注入正文，按之前拿到的内容执行。"
        return Command(update={"messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)]})

    body = SKILLS[name]["body"]
    return Command(
        update={
            "messages": [ToolMessage(body, tool_call_id=runtime.tool_call_id)],
            "loaded_skills": [name],
        }
    )


@tool
def check_orders(order_ids: list[str], aspect: str, runtime: ToolRuntime) -> Command:
    """一次核对多个订单，每个订单派一个隔离的子 agent 去查，互不干扰、并行执行。
    order_ids 传订单号列表（比如 ["KL-778", "KL-901"]），至少两个才用这个工具——
    只查一个订单用 get_order/get_policy 就够了。aspect 是要核对的角度，一句话，
    比如"能不能改期"、"退款政策是什么"。"""
    ack = ToolMessage(
        f"已经并行核对 {len(order_ids)} 个订单，结果在后面那条消息里。",
        tool_call_id=runtime.tool_call_id,
        name="check_orders",
    )
    return Command(update={"messages": [ack], "order_ids": order_ids, "check_aspect": aspect})


BASE_TOOLS = [get_order, get_policy, cancel_order, remember_note]
TOOLS = [*BASE_TOOLS, search_faq, load_skill, check_orders]
