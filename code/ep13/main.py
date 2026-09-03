"""跑第 11 期：
    uv run python -m ep13.main wang t1 "订单 KL-901 想改期，但已经超过免费改期窗口了，我这边突然要住院"
    uv run python -m ep13.main --resume <user_id> <thread_id> approve
    uv run python -m ep13.main --history <thread_id>
    uv run python -m ep13.main --memory <user_id>

每条 [agent] 回答后面 (prompt_tokens=NNN) 是这一轮真实发给模型的 token 数。
配了 Langfuse（见 common/settings.py）时，每次调用最后会打印这次的 trace id，
用 LANGFUSE_HOST 拼出链接就能直接打开看。
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command

from common import settings
from ep13.graph import build_graph
from ep13.mcp_client import load_mcp_tools
from ep13.tools import TOOLS

DATA = Path(__file__).parent / "data"
CHECKPOINT_DB = DATA / "checkpoints.sqlite"
MEMORY_DB = DATA / "memory.sqlite"


def build_run_config(thread_id: str, user_id: str) -> dict:
    """跟第 4 期起的 config 长得一样，多两样：callbacks 挂上 Langfuse 的
    handler（没配 key 就是空列表，跟没这回事一样）；metadata 里那两个
    langfuse_ 开头的字段，是 Langfuse 认的专用名字，用来把这次调用产生
    的 trace 归到正确的会话和用户名下——不这么传，Langfuse 后台会把所有
    对话的 trace 全堆在一起，看不出哪条是哪场对话的。"""
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    if settings.LANGFUSE_ENABLED:
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]
        config["metadata"] = {"langfuse_session_id": thread_id, "langfuse_user_id": user_id}
    return config


def report_trace(config: dict) -> None:
    """打印这次调用的 trace id，并且在进程退出前把它送到 Langfuse——CLI
    跑完就退出，Langfuse 的 SDK 默认攒够一批或者等一段时间才发送，
    `flush()` 强制立刻发，晚一步这条记录就随着进程一起消失了。"""
    callbacks = config.get("callbacks") or []
    if not callbacks:
        return
    from langfuse import get_client

    trace_id = callbacks[0].last_trace_id
    get_client().flush()
    if trace_id:
        print(f"[langfuse] trace: {settings.LANGFUSE_HOST}/trace/{trace_id}")


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
            config = build_run_config(thread_id, user_id)
            await print_stream(graph.astream(Command(resume=decision), config=config, stream_mode="updates"))
            report_trace(config)
            return

        user_id, thread_id, question = args[0], args[1], " ".join(args[2:])
        config = build_run_config(thread_id, user_id)
        state = {"messages": [HumanMessage(question)]}
        await print_stream(graph.astream(state, config=config, stream_mode="updates"))
        report_trace(config)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
