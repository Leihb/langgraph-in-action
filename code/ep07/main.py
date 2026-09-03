"""跑第 7 期（MCP 工具是异步的，这一期起 main.py 换成 asyncio，checkpointer
和 store 也跟着换成 Async 版本——同步版在事件循环里直接报错，不是能不能用
的问题）：
    uv run python -m ep07.main wang t1 "现在东京几点，我想问问对方方不方便接电话"
    uv run python -m ep07.main wang t1 "帮我查一下 KL-778 的改期政策"
    uv run python -m ep07.main --resume <user_id> <thread_id> approve
    uv run python -m ep07.main --history <thread_id>
    uv run python -m ep07.main --memory <user_id>
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command

from ep07.graph import build_graph
from ep07.mcp_client import load_mcp_tools
from ep07.tools import TOOLS

CHECKPOINT_DB = Path(__file__).parent / "data" / "checkpoints.sqlite"
MEMORY_DB = Path(__file__).parent / "data" / "memory.sqlite"


async def print_stream(stream) -> None:
    async for update in stream:
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            print(f"[需要人工确认] {info.value}")
            print("  用 --resume <user_id> <thread_id> approve 或 reject 继续")
            continue
        for node, changed in update.items():
            for msg in changed["messages"]:
                if node == "agent" and msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"[agent] 要调 {call['name']}({call['args']})")
                elif node == "agent":
                    print(f"[agent] 回答：{msg.content}")
                else:
                    print(f"[tools] {msg.name} 返回：{str(msg.content)[:80]}")


async def show_history(graph, config) -> None:
    snapshot = await graph.aget_state(config)
    msgs = snapshot.values.get("messages", [])
    print(f"thread {config['configurable']['thread_id']}：{len(msgs)} 条消息，"
          f"下一步待执行节点：{snapshot.next or '(无，对话已结束)'}")
    for m in msgs:
        kind = type(m).__name__.replace("Message", "")
        text = m.content if m.content else f"tool_calls={[c['name'] for c in m.tool_calls]}"
        print(f"  {kind:<6} {str(text)[:60]}")


async def show_memory(store, user_id: str) -> None:
    item = await store.aget((user_id, "memory"), "note")
    print(f"user {user_id} 的笔记：{item.value['text'] if item else '(还没有记过东西)'}")


async def main_async() -> None:
    args = sys.argv[1:]
    if len(args) < 1:
        print(__doc__)
        raise SystemExit(1)

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver, \
            AsyncSqliteStore.from_conn_string(str(MEMORY_DB)) as store:
        await store.setup()

        if args[0] == "--memory":
            await show_memory(store, args[1])
            return

        if args[0] == "--history":
            # 看历史不用调工具，不用起 MCP 服务器进程，图只挂手写工具就够了
            graph = build_graph(saver, store, TOOLS)
            await show_history(graph, {"configurable": {"thread_id": args[1]}})
            return

        # 只有真的要跑一轮对话才连 MCP 服务器——起子进程有开销，能不连就不连
        mcp_tools = await load_mcp_tools()
        graph = build_graph(saver, store, TOOLS + mcp_tools)

        if args[0] == "--resume":
            user_id, thread_id, decision = args[1], args[2], args[3]
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            await print_stream(graph.astream(Command(resume=decision), config=config, stream_mode="updates"))
            return

        user_id, thread_id, question = args[0], args[1], " ".join(args[2:])
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        state = {"messages": [HumanMessage(question)]}
        await print_stream(graph.astream(state, config=config, stream_mode="updates"))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
