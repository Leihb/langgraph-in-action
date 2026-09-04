"""例子 4：多源知识库路由。官方 "router / multi-source knowledge base" 例子的重做。

    classify（模型：这个问题该问哪几个来源、各自问什么）
      → Send 并行派给 wiki / tickets / chat（各一个小 agent）
      → synthesize（模型：把几份汇报合成一个回答）

跟第 10 期的 Send 扇出是同一个机制，差别在"扇出几份、发给谁"由谁定：第 10 期是
状态里订单号列表的长度（代码定）；这里是分类那一步模型的判断——它要决定哪些来源
跟问题有关、每个来源该问什么子问题。分类是模型的活，之后的并行、汇合、隔离是图的活。
"""

import time
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import chat_model
from ex04_router_knowledge_base.sources import AGENTS, SOURCE_DESCRIPTIONS
from ex04_router_knowledge_base.state import RouterState, SourceInput

llm = chat_model()
classifier = llm.with_structured_output(
    {"title": "Routing", "type": "object",
     "properties": {"classifications": {"type": "array", "items": {"type": "object", "properties": {
         "source": {"type": "string", "enum": ["wiki", "tickets", "chat"]},
         "query": {"type": "string"}}, "required": ["source", "query"]}}},
     "required": ["classifications"]},
    method="json_mode",
)

CLASSIFY = """有三个知识来源：
{sources}

用户的问题：{question}

判断这个问题该去哪几个来源找答案——可以是一个、两个或三个，跟问题无关的来源不要选。
给每个选中的来源写一个针对它的子问题（这个来源里该找什么），比原问题更具体。

只输出一个 JSON 对象：{{"classifications": [{{"source": "wiki|tickets|chat", "query": "子问题"}}]}}"""

SYNTHESIZE = """用户的问题：{question}

各来源的汇报：
{reports}

综合上面的汇报，用中文回答用户。规则优先引用文档，具体案例引用工单，同事经验引用聊天记录，
每条结论后面标出处。汇报里没有的不要编；如果各来源都没找到，直说。"""


def classify(state: RouterState) -> dict:
    out = classifier.invoke(CLASSIFY.format(
        sources="\n".join(f"- {k}：{v}" for k, v in SOURCE_DESCRIPTIONS.items()), question=state["question"]))
    picks = [c for c in out.get("classifications", []) if c.get("source") in AGENTS and c.get("query")]
    return {"classifications": picks}


def route(state: RouterState) -> list[Send] | Literal["synthesize"]:
    """条件边：返回一组 Send，每个 Send 带着自己的输入去一个来源节点。选了零个来源就直接去汇总。"""
    if not state["classifications"]:
        return "synthesize"
    return [Send(c["source"], {"source": c["source"], "query": c["query"]}) for c in state["classifications"]]


def make_source_node(source: str):
    def node(inp: SourceInput) -> dict:
        t0 = time.monotonic()
        print(f"  [{source}] 开始：{inp['query']}")
        result = AGENTS[source].invoke({"messages": [HumanMessage(inp["query"])]})
        report = result["messages"][-1].content
        secs = round(time.monotonic() - t0, 2)
        print(f"  [{source}] 结束（{secs}s）")
        return {"results": [{"source": source, "result": report, "seconds": secs}]}
    return node


def synthesize(state: RouterState) -> dict:
    results = state.get("results", [])
    if not results:
        return {"final_answer": "这个问题跟三个知识来源都不相关，我这里查不到。"}
    reports = "\n\n".join(f"【{r['source']}】\n{r['result']}" for r in results)
    reply = llm.invoke(SYNTHESIZE.format(question=state["question"], reports=reports))
    return {"final_answer": reply.content.strip()}


def build_graph():
    builder = StateGraph(RouterState)
    builder.add_node("classify", classify)
    for source in AGENTS:
        builder.add_node(source, make_source_node(source))
        builder.add_edge(source, "synthesize")
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route, [*AGENTS, "synthesize"])
    builder.add_edge("synthesize", END)
    return builder.compile()


graph = build_graph()
