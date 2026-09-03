"""跑第 10 期：
    uv run python -m ep10.main wang t1 "帮我一起查一下 KL-778、KL-901、KL-315 这三个订单能不能改期"
    uv run python -m ep10.main wang t1 "订单 KL-901 想改期，但已经超过免费改期窗口了，我这边突然要住院"
    uv run python -m ep10.main --resume <user_id> <thread_id> approve
    uv run python -m ep10.main --history <thread_id>
    uv run python -m ep10.main --memory <user_id>

每条 [agent] 回答后面 (prompt_tokens=NNN) 是这一轮真实发给模型的 token 数。
`[lookup_order]` 那几行打的是子图开始/结束的时刻和耗时，用来看它们是不是真的
在同时跑，不是排队一个个查。
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command

from ep10.graph import build_graph
from ep10.mcp_client import load_mcp_tools
from ep10.tools import TOOLS

DATA = Path(__file__).parent / "data"
CHECKPOINT_DB = DATA / "checkpoints.sqlite"
MEMORY_DB = DATA / "memory.sqlite"


async def print_stream(stream) -> None:
    async for update in stream:
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            print(f"[需要人工确认] {info.value}")
            print("  用 --resume <user_id> <thread_id> approve 或 reject 继续")
            continue
        for node, changed in update.items():
            # 一次并行调用里只要有一个工具返回 Command，"tools" 这个节点的更新
            # 就不再是单个 {"messages": [...]} 字典，而是一份 [{"messages": [...]},
            # {"loaded_skills": [...]}, ...] 这样的列表——每个工具各自的写入分开列出。
            parts = changed if isinstance(changed, list) else [changed]
            for part in parts:
                for msg in part.get("messages", []):
                    if node == "agent" and msg.tool_calls:
                        usage = msg.usage_metadata or {}
                        tokens = usage.get("input_tokens")
                        suffix = f" (prompt_tokens={tokens})" if tokens is not None else ""
                        for call in msg.tool_calls:
                            print(f"[agent] 要调 {call['name']}({call['args']}){suffix}")
                    elif node == "agent":
                        usage = msg.usage_metadata or {}
                        tokens = usage.get("input_tokens")
                        suffix = f" (prompt_tokens={tokens})" if tokens is not None else ""
                        print(f"[agent] 回答：{msg.content}{suffix}")
                    else:
                        label = getattr(msg, "name", None) or type(msg).__name__
                        print(f"[{node}] {label} 返回：{str(msg.content)[:100]}")


async def show_history(graph, config) -> None:
    snapshot = await graph.aget_state(config)
    msgs = snapshot.values.get("messages", [])
    print(f"thread {config['configurable']['thread_id']}：{len(msgs)} 条消息，"
          f"已加载 skill：{snapshot.values.get('loaded_skills', [])}，"
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
            graph = build_graph(saver, store, TOOLS)
            await show_history(graph, {"configurable": {"thread_id": args[1]}})
            return

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
