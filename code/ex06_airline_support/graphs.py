"""例子 6：航空客服，四版图。官方 "Build a Customer Support Bot" 教程的重做。

- v1 零样本：一个助理、全部工具，想调就调。
- v2 每次工具前确认：多一个 fetch_user_info 节点先把客人机票放进提示词；任何工具调用前停下来等人。
- v3 只对写操作确认：工具分安全（查）和敏感（订/改/取消）两组，只有敏感组前面有闸门。
- v4 专项助理：主助理只查信息和转交，四个专项助理各管一个领域，dialog_state 栈记着现在谁在接待。

官方用编译期的 interrupt_before=["tools"] 停图，这本书从第 5 期起统一用节点内 interrupt()：
这里是一个 approve 节点，放在敏感工具节点前面——它把待执行的调用摆出来等人批，批了去
执行，不批就给每个调用回一条"被拒绝"的 ToolMessage 让助理重新想。
"""

from datetime import datetime
from typing import Annotated, Callable, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from common.llm import chat_model
from ex06_airline_support import prompts, tools

llm = chat_model()


# ---------- 状态 ----------

def update_dialog_stack(left: list[str], right: str | None) -> list[str]:
    """dialog_state 的 reducer：进专项助理时 push 一个名字，交回主助理时传 "pop"。"""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[list[Literal["update_flight", "book_car_rental", "book_hotel", "book_excursion"]], update_dialog_stack]


# ---------- 公用件 ----------

class Assistant:
    """一个提示词 + 一组工具 = 一个助理。模型偶尔会返回空内容，官方的处理是再催一句。"""

    def __init__(self, system: str, tool_list: list):
        self.system = system
        self.model = llm.bind_tools(tool_list)

    def __call__(self, state: State) -> dict:
        sys = SystemMessage(self.system.format(user_info=state.get("user_info", ""), time=datetime.now().isoformat(timespec="minutes")))
        messages = list(state["messages"])
        while True:
            result: AIMessage = self.model.invoke([sys, *messages])
            if not result.tool_calls and not (result.content or "").strip():
                messages = messages + [("user", "请给出实际的回答。")]
            else:
                return {"messages": [result]}


def fetch_user_info(state: State, config: RunnableConfig) -> dict:
    pid = config["configurable"]["passenger_id"]
    return {"user_info": str(tools.user_flights(pid))}


def make_approve(next_node: str, back_to: str) -> Callable:
    """闸门节点：把助理刚提出的工具调用摆给人看。批准去 next_node 执行；拒绝就给每个调用
    回一条 ToolMessage 说明原因，回到 back_to 让助理重新想。interrupt 在第一行。"""

    def approve(state: State) -> Command:
        calls = state["messages"][-1].tool_calls
        decision = interrupt({
            "pending": [{"tool": c["name"], "args": c["args"]} for c in calls],
            "hint": "回复 y 批准执行；回复别的内容视为拒绝，内容会作为原因转给助理",
        })
        if str(decision).strip().lower() in ("y", "yes", "approve", "批准", "同意"):
            return Command(goto=next_node)
        denied = [ToolMessage(content=f"用户拒绝了这次操作。原因：'{decision}'。请据此继续帮助用户。",
                              tool_call_id=c["id"]) for c in calls]
        return Command(update={"messages": denied}, goto=back_to)

    return approve


def tool_node(tool_list: list) -> ToolNode:
    return ToolNode(tool_list, handle_tool_errors="工具报错：{error}。请修正后重试。")


# ---------- v1：零样本 ----------

def build_v1(checkpointer):
    b = StateGraph(State)
    b.add_node("assistant", Assistant(prompts.PRIMARY_SINGLE, tools.ALL_TOOLS))
    b.add_node("tools", tool_node(tools.ALL_TOOLS))
    b.add_edge(START, "assistant")
    b.add_conditional_edges("assistant", tools_condition)
    b.add_edge("tools", "assistant")
    return b.compile(checkpointer=checkpointer)


# ---------- v2：每次工具前确认 ----------

def build_v2(checkpointer):
    b = StateGraph(State)
    b.add_node("fetch_user_info", fetch_user_info)
    b.add_node("assistant", Assistant(prompts.PRIMARY_SINGLE, tools.ALL_TOOLS))
    b.add_node("approve", make_approve("tools", "assistant"))
    b.add_node("tools", tool_node(tools.ALL_TOOLS))
    b.add_edge(START, "fetch_user_info")
    b.add_edge("fetch_user_info", "assistant")
    b.add_conditional_edges("assistant", tools_condition, {"tools": "approve", END: END})
    b.add_edge("tools", "assistant")
    return b.compile(checkpointer=checkpointer)


# ---------- v3：只对写操作确认 ----------

SENSITIVE_NAMES = {t.name for t in tools.SENSITIVE_TOOLS}


def route_v3(state: State) -> str:
    if tools_condition(state) == END:
        return END
    calls = state["messages"][-1].tool_calls
    return "approve" if any(c["name"] in SENSITIVE_NAMES for c in calls) else "safe_tools"


def build_v3(checkpointer):
    b = StateGraph(State)
    b.add_node("fetch_user_info", fetch_user_info)
    b.add_node("assistant", Assistant(prompts.PRIMARY_SINGLE, tools.ALL_TOOLS))
    b.add_node("safe_tools", tool_node(tools.SAFE_TOOLS))
    b.add_node("approve", make_approve("sensitive_tools", "assistant"))
    b.add_node("sensitive_tools", tool_node(tools.SENSITIVE_TOOLS))
    b.add_edge(START, "fetch_user_info")
    b.add_edge("fetch_user_info", "assistant")
    b.add_conditional_edges("assistant", route_v3, ["safe_tools", "approve", END])
    b.add_edge("safe_tools", "assistant")
    b.add_edge("sensitive_tools", "assistant")
    return b.compile(checkpointer=checkpointer)


# ---------- v4：主助理 + 四个专项助理 ----------

class CompleteOrEscalate(BaseModel):
    """标记当前任务已完成，或把对话控制权交回主助理（客人改主意、需要别的帮助时）。"""

    cancel: bool = True
    reason: str


class ToFlightBookingAssistant(BaseModel):
    """把工作转交给处理航班改签和取消的专项助理。"""

    request: str = Field(description="改签助理开始前需要跟客人确认的问题或已知的需求")


class ToBookCarRental(BaseModel):
    """把工作转交给处理租车预订的专项助理。"""

    location: str = Field(description="租车地点")
    start_date: str = Field(description="起租日期")
    end_date: str = Field(description="还车日期")
    request: str = Field(description="客人关于租车的其他要求")


class ToHotelBookingAssistant(BaseModel):
    """把工作转交给处理酒店预订的专项助理。"""

    location: str = Field(description="酒店所在城市")
    checkin_date: str = Field(description="入住日期")
    checkout_date: str = Field(description="退房日期")
    request: str = Field(description="客人关于酒店的其他要求")


class ToBookExcursion(BaseModel):
    """把工作转交给处理景点和活动推荐的专项助理。"""

    location: str = Field(description="游览地点")
    request: str = Field(description="客人关于景点/活动的其他要求")


SPECIALISTS = {
    # 名字: (提示词, 安全工具, 敏感工具, 转交工具类)
    "update_flight": (prompts.FLIGHT, tools.FLIGHT_SAFE, tools.FLIGHT_SENSITIVE, ToFlightBookingAssistant),
    "book_car_rental": (prompts.CAR, tools.CAR_SAFE, tools.CAR_SENSITIVE, ToBookCarRental),
    "book_hotel": (prompts.HOTEL, tools.HOTEL_SAFE, tools.HOTEL_SENSITIVE, ToHotelBookingAssistant),
    "book_excursion": (prompts.TRIP, tools.TRIP_SAFE, tools.TRIP_SENSITIVE, ToBookExcursion),
}
LABELS = {"update_flight": "航班改签助理", "book_car_rental": "租车助理", "book_hotel": "酒店预订助理", "book_excursion": "景点推荐助理"}


def make_entry(name: str) -> Callable:
    """进专项助理：回一条 ToolMessage 把主助理的转交调用配对上，同时 push dialog_state。"""

    def entry(state: State) -> dict:
        call_id = state["messages"][-1].tool_calls[0]["id"]
        return {"messages": [ToolMessage(content=prompts.ENTRY.format(assistant_name=LABELS[name]), tool_call_id=call_id)],
                "dialog_state": name}

    return entry


def make_specialist_router(name: str, safe: list) -> Callable:
    safe_names = {t.name for t in safe}

    def route(state: State) -> str:
        if tools_condition(state) == END:
            return END
        calls = state["messages"][-1].tool_calls
        if any(c["name"] == CompleteOrEscalate.__name__ for c in calls):
            return "leave_skill"
        return f"{name}_safe_tools" if all(c["name"] in safe_names for c in calls) else f"{name}_approve"

    return route


def leave_skill(state: State) -> dict:
    """交回主助理：pop 栈，并把 CompleteOrEscalate 那次调用配上一条 ToolMessage。"""
    msgs = []
    if state["messages"][-1].tool_calls:
        msgs.append(ToolMessage(content=prompts.RESUME_HOST, tool_call_id=state["messages"][-1].tool_calls[0]["id"]))
    return {"dialog_state": "pop", "messages": msgs}


def route_primary(state: State) -> str:
    if tools_condition(state) == END:
        return END
    name = state["messages"][-1].tool_calls[0]["name"]
    for key, (_, _, _, cls) in SPECIALISTS.items():
        if name == cls.__name__:
            return f"enter_{key}"
    return "primary_tools"


def route_to_workflow(state: State) -> str:
    """每轮开头：栈里有专项助理就直接去它那里，客人的话不用再经过主助理。"""
    stack = state.get("dialog_state") or []
    return stack[-1] if stack else "primary_assistant"


def build_v4(checkpointer):
    b = StateGraph(State)
    b.add_node("fetch_user_info", fetch_user_info)
    b.add_edge(START, "fetch_user_info")

    for name, (prompt, safe, sensitive, _) in SPECIALISTS.items():
        b.add_node(f"enter_{name}", make_entry(name))
        b.add_node(name, Assistant(prompt, safe + sensitive + [CompleteOrEscalate]))
        b.add_node(f"{name}_safe_tools", tool_node(safe))
        b.add_node(f"{name}_approve", make_approve(f"{name}_sensitive_tools", name))
        b.add_node(f"{name}_sensitive_tools", tool_node(sensitive))
        b.add_edge(f"enter_{name}", name)
        b.add_edge(f"{name}_safe_tools", name)
        b.add_edge(f"{name}_sensitive_tools", name)
        b.add_conditional_edges(name, make_specialist_router(name, safe),
                                [f"{name}_safe_tools", f"{name}_approve", "leave_skill", END])

    b.add_node("leave_skill", leave_skill)
    b.add_edge("leave_skill", "primary_assistant")

    primary_tools = [tools.search_flights, tools.lookup_policy]
    b.add_node("primary_assistant", Assistant(prompts.PRIMARY_ROUTER, primary_tools + [cls for *_, cls in SPECIALISTS.values()]))
    b.add_node("primary_tools", tool_node(primary_tools))
    b.add_conditional_edges("primary_assistant", route_primary, [*(f"enter_{k}" for k in SPECIALISTS), "primary_tools", END])
    b.add_edge("primary_tools", "primary_assistant")
    b.add_conditional_edges("fetch_user_info", route_to_workflow, ["primary_assistant", *SPECIALISTS])
    return b.compile(checkpointer=checkpointer)


BUILDERS = {1: build_v1, 2: build_v2, 3: build_v3, 4: build_v4}
