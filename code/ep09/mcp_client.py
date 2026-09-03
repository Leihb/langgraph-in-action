"""MCP 客户端：读 mcp_servers.yaml，连上配置的服务器，拿到工具列表。

这份文件不知道、也不需要知道 time 服务器内部怎么实现——MCP 的握手和调用
方式是标准化的，`MultiServerMCPClient` 负责把对方声明的工具转成
LangChain 认识的 `BaseTool`，接口跟我们自己写的 `@tool` 一样。
"""

from pathlib import Path

import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG = Path(__file__).parent / "mcp_servers.yaml"


async def load_mcp_tools() -> list:
    servers = yaml.safe_load(CONFIG.read_text())["servers"]
    client = MultiServerMCPClient(servers)
    # 一个服务器逐个连——一次性 client.get_tools() 全拿的话，只要有一个服务器
    # 连不上，会连累其他连得上的服务器一个工具都用不了。
    tools = []
    for name in servers:
        try:
            tools += await client.get_tools(server_name=name)
        except Exception as e:
            print(f"[mcp] 服务器 {name!r} 连不上，跳过：{type(e).__name__}: {e}")
    return tools
