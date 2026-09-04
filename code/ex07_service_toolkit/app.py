"""例子 7：一个"完整"的 LangGraph + FastAPI 服务，对照开源项目 agent-service-toolkit。

    uv run uvicorn ex07_service_toolkit.app:app --port 8000

第 12-14 期手搭的服务只托管一个 agent、四种 SSE 事件、resume 走单独的路由。这一篇把
agent-service-toolkit 那套接口搬过来，看一个"给别人用的通用服务"还差哪几样：

    GET  /info                      有哪些 agent、默认哪个
    POST /{agent_id}/invoke         一问一答，返回最后一条消息（停在 interrupt 就返回 interrupt 的内容）
    POST /{agent_id}/stream         SSE：中间消息 + 可选的 token 级流式（stream_tokens）
    POST /{agent_id}/history        按 thread_id 取整段对话
    POST /feedback                  给某次运行打分，写到 Langfuse（toolkit 写的是 LangSmith）
    GET  /health

跟第 12 期最大的两处差别：①中断不用单独的 /resume——每次请求先看这个 thread 是不是停在
interrupt 上，是就把这条消息当 resume 的答复；②流式多了 token 级别，`stream_mode` 同时
要 "updates" 和 "messages" 两种。
"""

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from common import settings
from ex07_service_toolkit.agents import AGENTS, DEFAULT_AGENT

DATA = Path(__file__).parent / "data"
bearer = HTTPBearer(auto_error=False)


def check_auth(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if not creds or creds.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="密钥不对")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    DATA.mkdir(exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(DATA / "checkpoints.sqlite")) as saver, \
            AsyncSqliteStore.from_conn_string(str(DATA / "store.sqlite")) as store:
        await store.setup()
        app.state.graphs = {}
        for key, spec in AGENTS.items():
            try:
                app.state.graphs[key] = await spec.build(saver, store)
                print(f"[agents] 已加载 {key}")
            except Exception as e:  # noqa: BLE001 — 一个 agent 起不来不该拖垮别的
                print(f"[agents] {key} 加载失败，跳过：{type(e).__name__}: {e}")
        yield


app = FastAPI(lifespan=lifespan)


# ---------- 对外的数据结构（照 toolkit 的 schema 精简） ----------

class UserInput(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str | None = None


class StreamInput(UserInput):
    stream_tokens: bool = True


class ChatMessage(BaseModel):
    type: str                       # human / ai / tool
    content: str
    tool_calls: list[dict] = []
    run_id: str | None = None


class Feedback(BaseModel):
    run_id: str = Field(description="invoke/stream 返回的 run_id（就是 Langfuse 的 trace id）")
    key: str = "human-feedback-stars"
    score: float
    comment: str | None = None


class HistoryInput(BaseModel):
    thread_id: str


def to_chat_message(m: BaseMessage) -> ChatMessage:
    content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
    return ChatMessage(type=m.type, content=content,
                       tool_calls=[{"name": c["name"], "args": c["args"]} for c in getattr(m, "tool_calls", None) or []])


def get_graph(request: Request, agent_id: str):
    graph = request.app.state.graphs.get(agent_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"没有叫 {agent_id!r} 的 agent，/info 看有哪些")
    return graph


# ---------- 核心：把一条用户消息变成图的输入 ----------

async def prepare(graph, inp: UserInput) -> tuple[dict[str, Any], str]:
    """thread_id / user_id 没给就生成；挂 Langfuse；停在 interrupt 上的 thread，这条消息当 resume。
    返回 (ainvoke/astream 的 kwargs, run_id)。"""
    thread_id = inp.thread_id or uuid.uuid4().hex[:8]
    user_id = inp.user_id or "anonymous"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    handler = None
    if settings.LANGFUSE_ENABLED:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        config["callbacks"] = [handler]
        config["metadata"] = {"langfuse_session_id": thread_id, "langfuse_user_id": user_id}
    snapshot = await graph.aget_state(config)
    interrupted = any(getattr(t, "interrupts", None) for t in snapshot.tasks)
    run_input = Command(resume=inp.message) if interrupted else {"messages": [HumanMessage(inp.message)]}
    return {"input": run_input, "config": config, "_handler": handler, "_thread": thread_id}, thread_id


def run_id_of(kwargs: dict) -> str | None:
    h = kwargs.get("_handler")
    return getattr(h, "last_trace_id", None) if h else None


# ---------- 路由 ----------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/info", dependencies=[Depends(check_auth)])
async def info(request: Request) -> dict:
    return {"agents": [{"key": k, "description": v.description, "loaded": k in request.app.state.graphs} for k, v in AGENTS.items()],
            "default_agent": DEFAULT_AGENT, "model": settings.MODEL_NAME}


@app.post("/{agent_id}/invoke", dependencies=[Depends(check_auth)])
async def invoke(agent_id: str, inp: UserInput, request: Request) -> ChatMessage:
    graph = get_graph(request, agent_id)
    kwargs, thread_id = await prepare(graph, inp)
    result = await graph.ainvoke(kwargs["input"], config=kwargs["config"])
    snapshot = await graph.aget_state(kwargs["config"])
    if snapshot.interrupts:
        out = ChatMessage(type="ai", content=json.dumps(snapshot.interrupts[0].value, ensure_ascii=False))
    else:
        out = to_chat_message(result["messages"][-1])
    out.run_id = run_id_of(kwargs)
    if settings.LANGFUSE_ENABLED:
        from langfuse import get_client
        get_client().flush()
    return out


async def sse(graph, inp: StreamInput) -> AsyncIterator[str]:
    kwargs, thread_id = await prepare(graph, inp)
    run_id_sent = False
    async for mode, event in graph.astream(kwargs["input"], config=kwargs["config"], stream_mode=["updates", "messages"]):
        if not run_id_sent:
            yield f"data: {json.dumps({'type': 'meta', 'thread_id': thread_id})}\n\n"
            run_id_sent = True
        if mode == "updates":
            for node, update in event.items():
                if node == "__interrupt__":
                    for it in update:
                        yield f"data: {json.dumps({'type': 'interrupt', 'content': it.value}, ensure_ascii=False)}\n\n"
                    continue
                parts = update if isinstance(update, list) else [update or {}]
                for part in parts:
                    for m in (part or {}).get("messages", []):
                        if isinstance(m, BaseMessage) and not (m.type == "human" and m.content == inp.message):
                            yield f"data: {json.dumps({'type': 'message', 'content': to_chat_message(m).model_dump()}, ensure_ascii=False)}\n\n"
        elif mode == "messages" and inp.stream_tokens:
            chunk, meta = event
            # 只有模型节点吐的 AIMessageChunk 才是 token；工具节点在这个流里也会冒消息，跳过
            if isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_calls:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'run_id': run_id_of(kwargs)})}\n\n"
    if settings.LANGFUSE_ENABLED:
        from langfuse import get_client
        get_client().flush()


@app.post("/{agent_id}/stream", dependencies=[Depends(check_auth)])
async def stream(agent_id: str, inp: StreamInput, request: Request) -> StreamingResponse:
    graph = get_graph(request, agent_id)
    return StreamingResponse(sse(graph, inp), media_type="text/event-stream")


@app.post("/{agent_id}/history", dependencies=[Depends(check_auth)])
async def history(agent_id: str, inp: HistoryInput, request: Request) -> dict:
    graph = get_graph(request, agent_id)
    snapshot = await graph.aget_state({"configurable": {"thread_id": inp.thread_id}})
    return {"thread_id": inp.thread_id, "messages": [to_chat_message(m).model_dump() for m in snapshot.values.get("messages", [])]}


@app.post("/feedback", dependencies=[Depends(check_auth)])
async def feedback(fb: Feedback) -> dict:
    """toolkit 写 LangSmith，这本书用 Langfuse：run_id 就是 trace id，打成一个 score 挂上去。"""
    if not settings.LANGFUSE_ENABLED:
        raise HTTPException(status_code=503, detail="没配 Langfuse，反馈无处可存")
    from langfuse import get_client

    lf = get_client()
    lf.create_score(trace_id=fb.run_id, name=fb.key, value=fb.score, comment=fb.comment)
    lf.flush()
    return {"status": "success"}
