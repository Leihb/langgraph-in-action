"""agent 注册表：一个服务托管几个 agent，按名字选。

agent-service-toolkit 的做法是一个字典：key 是 URL 里的 agent 名，值是描述加编译好的图。
这里挂两个这本书已有的：第 14 期那个客服 agent，例子 3 那个三步状态机。它们都吃
`messages`、都认 `thread_id`，所以能共用一套接口。

图在 lifespan 里编译（要等 checkpointer / store / MCP 连接就位），这里只登记"怎么建"。
"""

from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class AgentSpec:
    description: str
    build: Callable[..., Awaitable]  # async (saver, store) -> compiled graph


async def _build_support(saver, store):
    from ep14.graph import build_graph
    from ep14.mcp_client import load_mcp_tools
    from ep14.tools import TOOLS

    return build_graph(saver, store, TOOLS + await load_mcp_tools())


async def _build_state_machine(saver, store):
    from ex03_support_state_machine.agent import build_agent

    return build_agent(saver)


AGENTS: dict[str, AgentSpec] = {
    "support": AgentSpec("第 14 期的旅行客服 agent：查订单、政策、取消（需人工确认）、MCP 时间、skill、多订单并行", _build_support),
    "state-machine": AgentSpec("例子 3 的三步状态机客服：核实订单 → 诉求类型 → 给方案", _build_state_machine),
}
DEFAULT_AGENT = "support"
