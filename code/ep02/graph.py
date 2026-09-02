"""第 2 期：第一张图。三个节点串成一条线，模型只出现在其中两个里。"""

import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from common.llm import chat_model
from ep02.prompts import CATEGORIES, CLASSIFY, DRAFT
from ep02.state import DraftState

POLICIES = json.loads((Path(__file__).parent / "data" / "policies.json").read_text())
llm = chat_model()


def classify(state: DraftState) -> dict:
    """节点一：模型判断问题类别。只返回自己改动的字段。"""
    reply = llm.invoke(CLASSIFY.format(question=state["question"]))
    word = reply.content.strip().lower()
    category = word if word in CATEGORIES else "usage"
    return {"category": category, "trace": [f"classify -> {category}"]}


def lookup(state: DraftState) -> dict:
    """节点二：纯 Python 查政策，没有模型。"""
    product = POLICIES[state["product_id"]]
    policy = product[state["category"]]
    return {"policy": policy, "trace": [f"lookup -> {product['name']}"]}


def draft(state: DraftState) -> dict:
    """节点三：模型照着政策原文起草回复。"""
    name = POLICIES[state["product_id"]]["name"]
    reply = llm.invoke(
        DRAFT.format(name=name, policy=state["policy"], question=state["question"])
    )
    return {"draft": reply.content.strip(), "trace": ["draft -> done"]}


def build_graph():
    builder = StateGraph(DraftState)
    builder.add_node("classify", classify)
    builder.add_node("lookup", lookup)
    builder.add_node("draft", draft)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "lookup")
    builder.add_edge("lookup", "draft")
    builder.add_edge("draft", END)
    return builder.compile()


graph = build_graph()
