"""六个工具。改阶段的那几个返回 Command——同时写数据字段和 current_step。

这是这一篇的核心动作：**状态机的换挡由工具做**。模型决定"现在该调 record_issue 了"，
但"调完之后进入第三步"写在工具里，模型没法跳步，也没法忘了换挡。
"""

import json
from datetime import date, timedelta
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ex03_support_state_machine.state import SupportState

DATA = Path(__file__).parent / "data"
ORDERS = json.loads((DATA / "orders.json").read_text())
POLICIES = json.loads((DATA / "policies.json").read_text())


def _msg(text: str, runtime: ToolRuntime) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=runtime.tool_call_id)


@tool
def lookup_order(order_id: str, runtime: ToolRuntime[None, SupportState]) -> Command | str:
    """按订单号核实订单（形如 KL-778）。核实成功后自动进入下一步。"""
    order = ORDERS.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id}，请客人核对后再报一次"
    travel = (date.today() + timedelta(days=order["travel_in_days"])).isoformat()
    name = POLICIES[order["product_id"]]["name"]
    return Command(update={
        "messages": [_msg(f"订单核实成功：{order['customer']}，{name}，出行日期 {travel}", runtime)],
        "order_id": order_id, "customer": order["customer"], "product_id": order["product_id"],
        "product_name": name, "travel_date": travel,
        "current_step": "classify",
    })


@tool
def record_issue(issue_type: str, runtime: ToolRuntime[None, SupportState]) -> Command | str:
    """记录客人的诉求类型：reschedule（改期）、refund（退款）或 other。记录后进入给方案那一步。"""
    if issue_type not in ("reschedule", "refund", "other"):
        return f"issue_type 只能是 reschedule / refund / other，收到的是 {issue_type}"
    return Command(update={
        "messages": [_msg(f"已记录诉求类型：{issue_type}", runtime)],
        "issue_type": issue_type,
        "current_step": "resolve",
    })


@tool
def get_policy(topic: str, runtime: ToolRuntime[None, SupportState]) -> str:
    """查当前订单对应商品的政策原文。topic 只能是 reschedule / refund / usage。"""
    product = POLICIES[runtime.state["product_id"]]
    if topic not in ("reschedule", "refund", "usage"):
        return f"topic 只能是 reschedule / refund / usage，收到的是 {topic}"
    return f"{product['name']} 的 {topic} 政策：{product[topic]}"


@tool
def provide_solution(summary: str, runtime: ToolRuntime[None, SupportState]) -> Command:
    """把最终给客人的处理结论记下来（一两句话）。"""
    return Command(update={
        "messages": [_msg("结论已记录", runtime)],
        "solution": summary,
    })


@tool
def restart(reason: str, runtime: ToolRuntime[None, SupportState]) -> Command:
    """客人要换订单或改诉求时调用：清空已核实的订单信息，回到第一步重新核实。"""
    return Command(update={
        "messages": [_msg(f"已重置，回到核实订单这一步（原因：{reason}）", runtime)],
        "current_step": "identify",
        "order_id": "", "customer": "", "product_id": "", "product_name": "", "travel_date": "",
        "issue_type": "other", "solution": "",
    })


ALL_TOOLS = [lookup_order, record_issue, get_policy, provide_solution, restart]
