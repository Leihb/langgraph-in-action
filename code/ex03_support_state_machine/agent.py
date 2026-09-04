"""例子 3：客服状态机。官方 "handoffs / customer support" 那个例子的重做。

一个 agent 循环，分三步走：核实订单 → 搞清诉求 → 给方案。三步用的是同一个模型、
同一张图，换的只是每一步的系统提示词和它能看见的工具——由一个中间件在每次调
模型之前按 state 里的 `current_step` 换上。步子怎么往前走，写在工具里（tools.py）。

跟前面两个例子的差别：这里模型是在循环里自己决定下一步调什么工具的（第三档），
但它的选择范围每一步都被中间件收窄到两三个工具，走到哪一步由工具决定，模型跳不了。
"""

from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from common.llm import chat_model
from ex03_support_state_machine.prompts import STEP_CONFIG
from ex03_support_state_machine.state import SupportState
from ex03_support_state_machine.tools import ALL_TOOLS


def _step_request(request: ModelRequest) -> ModelRequest:
    """每次调模型之前跑一次：读 current_step，换提示词、换工具。"""
    step = request.state.get("current_step") or "identify"
    cfg = STEP_CONFIG[step]
    missing = [k for k in cfg["requires"] if not request.state.get(k)]
    if missing:
        # 上一步的工具没把该写的字段写进 state，这是代码 bug，不是模型的错——当场报出来
        raise RuntimeError(f"进入 {step} 阶段但 state 缺字段 {missing}")
    prompt = cfg["prompt"].format(**{**request.state, "today": date.today().isoformat()})
    tools = [t for t in request.tools if t.name in cfg["tools"]]
    return request.override(system_message=SystemMessage(prompt), tools=tools)


class StepConfigMiddleware(AgentMiddleware):
    """同步、异步两个入口都实现。只写同步版的话，用 `invoke()`/`stream()` 跑没问题，
    一旦这个 agent 被放进 FastAPI 用 `ainvoke()` 调，会直接报 NotImplementedError——
    例子 7 把它挂进服务时撞见的。"""

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return handler(_step_request(request))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return await handler(_step_request(request))


apply_step_config = StepConfigMiddleware()


def build_agent(checkpointer):
    return create_agent(
        chat_model(),
        tools=ALL_TOOLS,
        state_schema=SupportState,
        middleware=[apply_step_config],
        checkpointer=checkpointer,
    )
