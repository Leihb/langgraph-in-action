# LangGraph 加 FastAPI 的完整服务——对照 agent-service-toolkit

> 对照的是开源项目 JoshuaC215/agent-service-toolkit（4.5k★）：LangGraph + FastAPI + Streamlit
> 的一套完整服务模板。这一篇把它的 HTTP 接口搬到这本书的 agent 上，看第 12-14 期手搭的
> 那个服务还差哪几样。
> 用到的机制：第 12 期（FastAPI、SSE、鉴权）、第 13 期（存储）、第 5 期（interrupt）、第 11 期（Langfuse）。

第 12 到 14 期从零搭了一个服务并放到了公网上。它能用，但它是"这本书的服务"：只托管一个
agent，四种 SSE 事件是自己定的，恢复中断走一条单独的路由。agent-service-toolkit 是社区里
被拿来当起点最多的一个模板，它的接口设计经过很多人用过。这一篇不重写服务，把它的接口
搬过来，托管这本书已有的两个 agent，跑一遍，然后逐项对照：哪些是通用服务该有而我们
没做的，哪些是它有但这本书不需要的。

## 接口长什么样

| 路径 | 做什么 | 第 12 期有没有 |
|---|---|---|
| `GET /info` | 列出托管的 agent、默认哪个、用的模型 | 没有（只有一个 agent） |
| `POST /{agent_id}/invoke` | 一问一答，返回最后一条消息；停在 interrupt 就返回 interrupt 的内容 | `/chat` 只有流式 |
| `POST /{agent_id}/stream` | SSE，中间消息加可选的 token 级流式 | 有 SSE，没有 token 级 |
| `POST /{agent_id}/history` | 按 `thread_id` 取整段对话 | 没有 |
| `POST /feedback` | 给某次运行打分 | 没有 |
| `GET /health` | 健康检查 | 有 |

还有一条看不见的差别：**没有 `/resume`**。每次请求先看这个 thread 有没有停在 interrupt
上，是就把这条消息当 resume 的答复。第 12 期用两个路由区分，toolkit 用一个。

## 敲进去

代码在 `code/ex07_service_toolkit/`：`agents.py`（注册表）、`app.py`（服务）。没有新的 agent。

### 注册表：一个服务，几个 agent

```python
@dataclass
class AgentSpec:
    description: str
    build: Callable[..., Awaitable]  # async (saver, store) -> compiled graph


AGENTS: dict[str, AgentSpec] = {
    "support": AgentSpec("第 14 期的旅行客服 agent：……", _build_support),
    "state-machine": AgentSpec("例子 3 的三步状态机客服：……", _build_state_machine),
}
DEFAULT_AGENT = "support"
```

toolkit 的注册表是一个字典，key 是 URL 里的 agent 名，值是描述加图。这里挂两个这本书
已有的：第 14 期的客服 agent，例子 3 的状态机。它们都吃 `messages`、都认 `thread_id`，
所以能共用一套接口。图在 `lifespan` 里编译——要等 checkpointer、store、MCP 连接就位——
一个 agent 起不来只跳过它，不拖垮别的：

```python
for key, spec in AGENTS.items():
    try:
        app.state.graphs[key] = await spec.build(saver, store)
    except Exception as e:
        print(f"[agents] {key} 加载失败，跳过：{type(e).__name__}: {e}")
```

### 一条消息进来，先看要不要 resume

```python
async def prepare(graph, inp: UserInput) -> tuple[dict, str]:
    thread_id = inp.thread_id or uuid.uuid4().hex[:8]
    user_id = inp.user_id or "anonymous"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    if settings.LANGFUSE_ENABLED:
        handler = CallbackHandler()
        config["callbacks"] = [handler]
        config["metadata"] = {"langfuse_session_id": thread_id, "langfuse_user_id": user_id}
    snapshot = await graph.aget_state(config)
    interrupted = any(getattr(t, "interrupts", None) for t in snapshot.tasks)
    run_input = Command(resume=inp.message) if interrupted else {"messages": [HumanMessage(inp.message)]}
    ...
```

`aget_state` 看 thread 当前有没有停在 interrupt 上的任务。有，这条消息就是对 interrupt
的答复，包成 `Command(resume=...)`；没有，就是一条普通的新消息。客户端不用知道两种情况
的区别——第 5 期那个"停在 interrupt 时发普通消息会把对话搞坏"的坑，在这一层被挡掉了。

### 流式：两种模式同时开

```python
async for mode, event in graph.astream(kwargs["input"], config=kwargs["config"], stream_mode=["updates", "messages"]):
    if mode == "updates":
        ...  # 节点级：完整的消息、interrupt
    elif mode == "messages" and inp.stream_tokens:
        chunk, meta = event
        if isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_calls:
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"
```

`stream_mode` 传一个列表，每个事件带着它来自哪种模式。`"updates"` 是第 12 期用的那种，
一个节点跑完给一次；`"messages"` 是模型每吐一个 token 给一次。客户端两种都收：token 用来
边打字边显示，完整消息用来记录工具调用和最终回复。`stream_tokens=false` 就只收前者。

### 反馈：写到 Langfuse

```python
@app.post("/feedback")
async def feedback(fb: Feedback) -> dict:
    lf = get_client()
    lf.create_score(trace_id=fb.run_id, name=fb.key, value=fb.score, comment=fb.comment)
    lf.flush()
    return {"status": "success"}
```

toolkit 把反馈写到 LangSmith，这本书用 Langfuse，换一个 SDK 调用。`run_id` 就是这次运行
在 Langfuse 里的 trace id——`invoke`/`stream` 的返回里带着它（从 `CallbackHandler.last_
trace_id` 取），客户端拿它回来打分，分数挂在那条 trace 上，在第 11 期那个界面里能看到。

## 跑起来

```bash
cd code
export API_KEY=sk-ex07
uv run uvicorn ex07_service_toolkit.app:app --port 8000

curl http://localhost:8000/info -H "Authorization: Bearer sk-ex07"
curl -X POST http://localhost:8000/support/invoke -H "Authorization: Bearer sk-ex07" -H "Content-Type: application/json" \
  -d '{"message":"我的订单 KL-778 能改期吗","thread_id":"t1","user_id":"wang"}'
curl -N -X POST http://localhost:8000/support/stream ... -d '{"message":"那退款政策呢","thread_id":"t1","user_id":"wang","stream_tokens":true}'
```

## 你应该看到什么

### 两个 agent，一套接口

```
$ curl /info
{"agents":[{"key":"support","description":"第 14 期的旅行客服 agent：……","loaded":true},
           {"key":"state-machine","description":"例子 3 的三步状态机客服：……","loaded":true}],
 "default_agent":"support","model":"deepseek-v4-flash"}

$ curl -X POST /nope/invoke ...
{"detail":"没有叫 'nope' 的 agent，/info 看有哪些"}

$ curl -X POST /support/invoke -d '{"message":"我的订单 KL-778 能改期吗","thread_id":"t1","user_id":"wang"}'
{"type":"ai","content":"订单 KL-778（东京迪士尼一日票，出行日 2026-09-08）目前可以改期。……","run_id":"6ed649ef9509e36bcd3c40c125b35626"}

$ curl -X POST /state-machine/invoke -d '{"message":"订单 KL-315 想改期","thread_id":"s2"}'
{"type":"ai","content":"已为您记录"改期"诉求。请问您希望把这张票改到哪一天呢？……"}
```

同一个服务、同一个请求格式，URL 里换个名字就是另一个 agent。`run_id` 是 Langfuse 的
trace id。

### token 级流式

```
$ curl -N -X POST /support/stream -d '{"message":"那退款政策呢","thread_id":"t1","user_id":"wang","stream_tokens":true}'
data: {"type": "meta", "thread_id": "t1"}
data: {"type": "message", "content": {"type": "ai", "tool_calls": [{"name": "get_policy", "args": {"product_id": "SKU-1001", "topic": "refund"}}], ...}}
data: {"type": "message", "content": {"type": "tool", "content": "东京迪士尼一日票 的 refund 政策：出行日前 7 天可全额退款；7 天内不支持退款。", ...}}
data: {"type": "token", "content": "订单"}
data: {"type": "token", "content": " KL"}
……（一共 61 个 token 事件）
data: {"type": "message", "content": {"type": "ai", "content": "订单 KL-778 的退款政策：出行日前 7 天可全额退款，……", ...}}
data: {"type": "done", "run_id": "1edf6e0f7e6a8d693487e61679d1bcfb"}
```

一次请求：1 条 meta、3 条完整消息、61 个 token、1 条 done。工具调用和工具返回是完整
消息（`"updates"` 模式），最终回复先以 61 个 token 逐个到达（`"messages"` 模式），最后再
以一条完整消息到达。同一个 thread 上"那退款政策呢"没带订单号，模型接着上一轮的 KL-778
查——checkpointer 照常工作。

### 中断不用单独的路由

```
$ curl -X POST /support/invoke -d '{"message":"帮我取消订单 KL-901","thread_id":"t2","user_id":"chen"}'
{"type":"ai","content":"{\"action\": \"cancel_order\", \"order_id\": \"KL-901\", \"customer\": \"陈先生\", \"product\": \"东京迪士尼一日票\"}", ...}

$ curl -X POST /support/invoke -d '{"message":"approve","thread_id":"t2","user_id":"chen"}'
{"type":"ai","content":"您的订单 KL-901 已成功取消，……", ...}

$ curl -X POST /support/history -d '{"thread_id":"t2"}'
   human 帮我取消订单 KL-901
   ai    [{'name': 'cancel_order', 'args': {'order_id': 'KL-901'}}]
   tool  订单 KL-901 已取消
   ai    您的订单 KL-901 已成功取消，……
```

第一次 `invoke` 停在 interrupt，返回的是 interrupt 的内容（要审批的取消动作）。第二次还是
`invoke`，还是那个 thread，消息是 `approve`——服务自己判断出这是对 interrupt 的答复，
图从停下的地方接着跑。`history` 里四条消息，中间没有任何"resume"的痕迹。

### 反馈落到 Langfuse

```
$ curl -X POST /feedback -d '{"run_id":"82384a15…","key":"human-feedback-stars","score":0.8,"comment":"答得对"}'
{"status":"success"}

$ curl "$LANGFUSE_HOST/api/public/v3/scores?traceId=82384a15…"
[('human-feedback-stars', 0.8, ...)]
```

分数挂上去了。但这一条也要说：拿一个不存在的 `run_id` 打分，服务同样返回 `success`，
Langfuse 也照收——SDK 是异步批量发送的，不校验 trace 存在与否。toolkit 写 LangSmith 时
`create_feedback` 会同步报错。要严格，得在服务里先查一次 trace。

### 撞到的一个坑：同步中间件进不了异步服务

例子 3 那个状态机第一次挂上来，请求直接 500：

```
NotImplementedError: Asynchronous implementation of awrap_model_call is not available.
You are likely encountering this error because you defined only the sync version (wrap_model_call)
and invoked your agent in an asynchronous context (e.g., using `astream()` or `ainvoke()`).
```

例子 3 的中间件按官方文档用 `@wrap_model_call` 装饰器写，只有同步版本；命令行 `invoke()`
跑了几十轮都没事，进了 FastAPI 用 `ainvoke()` 一调就炸。修法是改成继承 `AgentMiddleware`、
同步异步各实现一个方法，两边共用同一段逻辑（例子 3 的代码和正文已经改过）。之后
`state-machine` 的 invoke 和 stream（42 个 token）都通了。

**凡是可能进服务的 agent，中间件一开始就按两个入口写。**

## 发生了什么

**通用服务和专用服务差的是"面向未知调用方"的那几样。** 第 12 期的服务假设调用方知道
这个 agent 是什么、知道停在 interrupt 时要调 `/resume`、知道四种事件各是什么。toolkit
的接口假设调用方什么都不知道：先 `/info` 看有什么，`invoke` 拿结果，`history` 拿上下文，
中断由服务自己处理，反馈有地方存。这些都不难做——这一篇加起来两百行——但它们决定了
一个服务能不能被一个没读过这本书的人接进自己的前端。

**`stream_mode` 是一个列表，这是 token 级流式的全部秘密。** 第 12 期只用 `"updates"`，
一个节点跑完给一次，所以客人看到的是一句话整块出现。加上 `"messages"`，模型每吐一个
token 就有一个事件。两种事件混在一条 SSE 里，用 `type` 字段区分，客户端各取所需。

**把 interrupt 的判断收进服务，客户端就少一个能出错的地方。** 第 5 期发现过：thread
停在 interrupt 上时，客户端如果发了一条普通消息而不是 resume，对话历史会坏掉、救不回来。
第 12 期把这个责任交给客户端（记得调 `/resume`）；这一篇服务每次先 `aget_state` 看一眼，
客户端发什么都行。

**run_id 把服务、观测、反馈串起来。** 每次运行返回一个 id，这个 id 在 Langfuse 里就是
那条 trace；客户端拿它打分，分数挂在 trace 上；第 15 期的评测也能按 trace 找到这次运行的
全部工具调用。三样东西用同一个 id 对齐，是 toolkit 设计里最值得抄的一点。

**toolkit 有而这一篇没做的。** `/threads`（按 `user_id` 列出这个人的所有对话，要在
checkpointer 的表上查）；每次请求指定模型；AG-UI 协议的路由；一个 Python 客户端库；
Streamlit 界面；Postgres/Mongo 切换（第 13 期做过 Postgres）；Docker Compose（第 14 期做过
Docker）。前四样是真实产品会要的，后三样这本书前面碰过。

## 常见问题

**为什么不直接用 agent-service-toolkit？** 可以用，它就是给人 fork 的。这一篇的价值是
知道它每个接口为什么存在——读完第 12-14 期再看它，能分清哪些是必需的、哪些是它作者的
偏好。直接 fork 也行，但改的时候会更有底。

**两个 agent 共用一个 checkpointer 文件，thread_id 会不会串？** 这一篇会：`thread_id` 是
调用方给的，两个 agent 用同一个 id 就写到同一条 checkpoint 上，状态结构不同会出错。toolkit
在 `metadata` 里记 `agent_id`，`/threads` 按它过滤。简单的隔离办法是每个 agent 一个
checkpointer 文件，或者 `thread_id` 前面拼上 agent 名。

**`invoke` 只返回最后一条消息，中间的工具调用丢了？** toolkit 的 `invoke` 也只返回最后
一条（它的代码注释说明了这一点）。要中间过程用 `stream`，或者用 `history`。

**token 级流式对所有节点都生效吗？** `"messages"` 模式对每一次模型调用都发 token，包括
中间那些决定调工具的调用——那些调用的 `content` 通常是空的、只有 `tool_calls`，代码里
过滤掉了。第 10 期那种子图里的模型调用也会发出来，要区分得看事件的 metadata。

**第 14 期那个线上服务要不要换成这套接口？** 那个已经删了（第 14 期正文说过）。如果要
再部一次，用这一篇的 `app.py` 加第 14 期的 Dockerfile 就行，两处改动：`CMD` 里的模块名，
以及 MCP 那个 `uvx` 首次启动要下载依赖——这一篇本机第一次起服务等了一分多钟就是它。

## 加分练习

1. 实现 `/threads`：按 `user_id` 列出这个人最近的对话。`checkpointer.alist(config, filter=...)`
   能按 metadata 过滤，先在 `prepare` 里把 `user_id` 和 `agent_id` 写进 `metadata`。
2. 给 `/feedback` 加一步校验：先用 Langfuse 的 API 查 trace 存在再打分，不存在返回 404。
3. 用第 15 期的 `runner.py` 对着这个服务（而非对着图）跑评测——task 函数里改成 HTTP 调用。
   评测的是"服务"而不是"图"时，多了哪些能出错的地方？
4. 把例子 5 的记忆 agent 也挂进注册表。它的 `call_model` 里 `store.search` 是同步调用，
   `AsyncSqliteStore` 会怎么反应？照例子 3 的经验改。
