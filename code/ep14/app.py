"""第 14 期：把第 13 期那个服务放到公网上，给别人一个能直接调的地址。

业务代码跟第 13 期一字不差——这一期动的全在代码外面：同目录的
`Dockerfile`（怎么把这个进程打进容器）、Render 上的一组环境变量（密钥
从哪来）、和一个免费的托管 Postgres（第 13 期讲的"存储要活在进程外面"
在这里不是选项，是免费层的硬约束：容器没有持久磁盘）。

本地照旧能跑：

    uv run uvicorn ep14.app:app --port 8000

容器方式（构建上下文是 code/ 这一层，见 Dockerfile 开头的说明）：

    docker build -f ep14/Dockerfile -t langgraph-ep14 .
    docker run -p 8000:8000 -e API_KEY=... -e POSTGRES_URL=... langgraph-ep14

线上那份不是从本机推镜像上去的，是 Render 从 GitHub 仓库自己构建的——
本机是 arm64、Render 是 amd64，本机构建的镜像上去跑不了，这条坑正文有讲。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from common import settings
from ep14.graph import build_graph
from ep14.mcp_client import load_mcp_tools
from ep14.tools import TOOLS

DATA = Path(__file__).parent / "data"
CHECKPOINT_DB = DATA / "checkpoints.sqlite"
MEMORY_DB = DATA / "memory.sqlite"

bearer = HTTPBearer()


def check_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> None:
    if creds.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="密钥不对")


async def _open_storage():
    """Postgres 和 SQLite 的 saver/store 都是 `from_conn_string()` 这同一套
    异步上下文管理器接口——第 4 期起用的 checkpointer 概念没变，变的只是
    连接串指向哪。切换成本几乎全部被这套接口封住了。"""
    if settings.POSTGRES_URL:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore

        saver_cm = AsyncPostgresSaver.from_conn_string(settings.POSTGRES_URL)
        store_cm = AsyncPostgresStore.from_conn_string(settings.POSTGRES_URL)
        saver = await saver_cm.__aenter__()
        store = await store_cm.__aenter__()
        await saver.setup()
        await store.setup()
        print(f"[storage] Postgres: {settings.POSTGRES_URL.split('@')[-1]}")
        return saver, store, saver_cm, store_cm

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    saver_cm = AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB))
    store_cm = AsyncSqliteStore.from_conn_string(str(MEMORY_DB))
    saver = await saver_cm.__aenter__()
    store = await store_cm.__aenter__()
    await store.setup()
    print(f"[storage] SQLite: {CHECKPOINT_DB}（没设 POSTGRES_URL，多进程 --workers 下不安全，见正文）")
    return saver, store, saver_cm, store_cm


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    saver, store, saver_cm, store_cm = await _open_storage()
    mcp_tools = await load_mcp_tools()
    app.state.graph = build_graph(saver, store, TOOLS + mcp_tools)
    app.state.tasks = {}  # thread_id -> asyncio.Task，防止后台任务被垃圾回收
    try:
        yield
    finally:
        await saver_cm.__aexit__(None, None, None)
        await store_cm.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    user_id: str
    thread_id: str
    decision: str


def build_run_config(thread_id: str, user_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    if settings.LANGFUSE_ENABLED:
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]
        config["metadata"] = {"langfuse_session_id": thread_id, "langfuse_user_id": user_id}
    return config


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def sse_events(graph, run_input, config: dict) -> AsyncIterator[str]:
    async for update in graph.astream(run_input, config=config, stream_mode="updates"):
        if "__interrupt__" in update:
            (info,) = update["__interrupt__"]
            yield _sse({"type": "interrupt", "payload": info.value})
            continue
        for node, changed in update.items():
            parts = changed if isinstance(changed, list) else [changed]
            for part in parts:
                for msg in part.get("messages", []):
                    if node == "agent" and msg.tool_calls:
                        for call in msg.tool_calls:
                            yield _sse({"type": "tool_call", "name": call["name"], "args": call["args"]})
                    elif node == "agent":
                        yield _sse({"type": "answer", "content": msg.content})
    yield _sse({"type": "done"})
    if settings.LANGFUSE_ENABLED:
        from langfuse import get_client

        get_client().flush()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(check_auth)])
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    graph = request.app.state.graph
    config = build_run_config(req.thread_id, req.user_id)
    state = {"messages": [HumanMessage(req.message)]}
    return StreamingResponse(sse_events(graph, state, config), media_type="text/event-stream")


@app.post("/chat/resume", dependencies=[Depends(check_auth)])
async def resume(req: ResumeRequest, request: Request) -> StreamingResponse:
    graph = request.app.state.graph
    config = build_run_config(req.thread_id, req.user_id)
    return StreamingResponse(
        sse_events(graph, Command(resume=req.decision), config), media_type="text/event-stream"
    )


@app.post("/chat/submit", dependencies=[Depends(check_auth)])
async def submit(req: ChatRequest, request: Request) -> dict:
    """跟 `/chat` 干的是同一件事，区别是不等结果——起一个后台任务，
    立刻把 thread_id 交回去。任务在哪个进程后台跑，结果就落在 Postgres
    里，跟客户端后面拿哪个进程去 `/status` 轮询没有关系。"""
    graph = request.app.state.graph
    config = build_run_config(req.thread_id, req.user_id)
    state = {"messages": [HumanMessage(req.message)]}
    task = asyncio.create_task(graph.ainvoke(state, config=config))
    request.app.state.tasks[req.thread_id] = task
    return {"thread_id": req.thread_id, "status": "submitted"}


@app.get("/chat/{thread_id}/status", dependencies=[Depends(check_auth)])
async def status(thread_id: str, user_id: str, request: Request) -> dict:
    """轮询用：不管提交那次任务是不是在这个进程里跑的，状态永远从
    Postgres 里现查——`graph.aget_state` 读的是 checkpointer 落盘的
    最新一条，不是某个进程内存里的变量。"""
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="没有这个 thread_id")
    interrupted = bool(snapshot.interrupts)
    done = not snapshot.next and not interrupted
    last = snapshot.values["messages"][-1]
    return {
        "thread_id": thread_id,
        "status": "interrupted" if interrupted else ("done" if done else "running"),
        "last_message": last.content if not getattr(last, "tool_calls", None) else None,
        "interrupt_payload": snapshot.interrupts[0].value if interrupted else None,
    }
