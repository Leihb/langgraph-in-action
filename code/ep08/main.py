"""跑第 8 期。两种检索方案在同一个入口里，第一个参数选方案：
    uv run python -m ep08.main tool wang t1 "机场大巴能带多大的行李箱"
    uv run python -m ep08.main node wang t1 "机场大巴能带多大的行李箱"
    uv run python -m ep08.main tool wang t1 "帮我查一下 KL-778 的改期政策"
    uv run python -m ep08.main --resume <mode> <user_id> <thread_id> approve
    uv run python -m ep08.main --history <mode> <thread_id>
    uv run python -m ep08.main --memory <user_id>
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command

from ep08 import graph_node, graph_tool
from ep08.mcp_client import load_mcp_tools
from ep08.tools import BASE_TOOLS, TOOLS

DATA = Path(__file__).parent / "data"
MEMORY_DB = DATA / "memory.sqlite"

# 两个方案各自的图构造函数 + 各自的工具列表：node 方案里 search_faq 不是
# 工具，agent 拿不到它，检索完全由 retrieve 节点包办。
BUILDERS = {
    "tool": (graph_tool.build_graph, TOOLS),
    "node": (graph_node.build_graph, BASE_TOOLS),
}


def checkpoint_db(mode: str) -> Path:
    # 两个方案的图结构不一样（node 方案多一个 retrieve 节点），checkpoint
    # 分开存，不共用一个 thread_id 空间。
    return DATA / f"checkpoints_{mode}.sqlite"


async def print_stream(stream) -> None:
    async for update in stream:
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            print(f"[需要人工确认] {info.value}")
            print("  用 --resume <mode> <user_id> <thread_id> approve 或 reject 继续")
            continue
        for node, changed in update.items():
            if node == "retrieve":
                print(f"[retrieve] 召回：\n{changed['retrieved']}")
                continue
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

    async with AsyncSqliteStore.from_conn_string(str(MEMORY_DB)) as store:
        await store.setup()

        if args[0] == "--memory":
            await show_memory(store, args[1])
            return

        if args[0] == "--history":
            mode, thread_id = args[1], args[2]
            build, tools = BUILDERS[mode]
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db(mode))) as saver:
                graph = build(saver, store, tools)  # 看历史不用调工具，随便传一份就够
                await show_history(graph, {"configurable": {"thread_id": thread_id}})
            return

        if args[0] == "--resume":
            mode, user_id, thread_id, decision = args[1], args[2], args[3], args[4]
            build, tools = BUILDERS[mode]
            mcp_tools = await load_mcp_tools()
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db(mode))) as saver:
                graph = build(saver, store, tools + mcp_tools)
                config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
                await print_stream(graph.astream(Command(resume=decision), config=config, stream_mode="updates"))
            return

        mode, user_id, thread_id, question = args[0], args[1], args[2], " ".join(args[3:])
        build, tools = BUILDERS[mode]
        mcp_tools = await load_mcp_tools()
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db(mode))) as saver:
            graph = build(saver, store, tools + mcp_tools)
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            state = {"messages": [HumanMessage(question)]}
            await print_stream(graph.astream(state, config=config, stream_mode="updates"))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
