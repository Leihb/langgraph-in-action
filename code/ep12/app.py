"""第 12 期：把第 11 期那个客服 agent 包进一个能给同事用的 HTTP 服务。

    uv run uvicorn ep12.app:app --port 8000

三件事，标题里都点了名：
- **流式**：`POST /chat` 返回 SSE，前端一收到 token/工具调用就能显示，不用等
  整轮跑完。
- **会话**：图和上一期一样接 `AsyncSqliteSaver`/`AsyncSqliteStore`，但只在
  进程启动时建一次连接，不是每次请求都开一次库文件——这是从 CLI 脚本换成
  常驻服务最容易漏掉的一步。
- **最简鉴权**：一个共享密钥挡在最外层，够教学用，生产该换成什么在"发生了
  什么"里说清楚。
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command
from pydantic import BaseModel

from common import settings
from ep12.graph import build_graph
from ep12.mcp_client import load_mcp_tools
from ep12.tools import TOOLS

DATA = Path(__file__).parent / "data"
CHECKPOINT_DB = DATA / "checkpoints.sqlite"
MEMORY_DB = DATA / "memory.sqlite"

bearer = HTTPBearer()


def check_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> None:
    if creds.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="密钥不对")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 进程活着的这段时间只开一次库文件、只连一次 MCP 服务器——每个请求
    # 复用同一份 graph，跟 CLI 版"每次调用重新连一次"是关键差别。
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver, \
            AsyncSqliteStore.from_conn_string(str(MEMORY_DB)) as store:
        await store.setup()
        mcp_tools = await load_mcp_tools()
        app.state.graph = build_graph(saver, store, TOOLS + mcp_tools)
        yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    user_id: str
    thread_id: str
    decision: str  # "approve" 或 "reject"


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
    """把 `astream(..., stream_mode="updates")` 那份内部结构，翻成前端能
    直接消费的几种事件：工具调用、模型的最终回答、需要人工确认、结束。
    事件形状故意跟第 3-11 期的内部结构不一样——那是图的实现细节，不应该
    透给调用方；调用方只应该认这几个事件类型，图内部怎么变都不影响它们。"""
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
