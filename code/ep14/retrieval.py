"""检索：本地 embedding 算相似度，不调用任何外部 API。

DeepSeek 走的是 chat 接口，不提供 embedding。这一期换成一个纯本地跑的
中文 embedding 模型（BAAI/bge-small-zh-v1.5），第一次跑会自动下载权重
（约 100MB），之后完全离线，不需要模型网关，也不需要 key。
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).parent / "data"
FAQ = json.loads((DATA / "faq.json").read_text())

# bge 系列对"查询"和"文档"用不同的前缀约定：查询侧要加这句提示，文档侧不加
# ——这是这个模型训练时定下的规矩，不是我们的设计，换模型要重新查对方文档。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model: SentenceTransformer | None = None
_faq_vecs: np.ndarray | None = None


def _encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        # 权重下载过一次之后本地就有缓存了，但默认加载方式每次还是会联网核对
        # 一遍版本——网络不稳的时候这一步会卡很久（亲测卡满两分钟）。
        # local_files_only=True 直接跳过这一步，缓存里没有才退回联网下载。
        try:
            _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except OSError:
            _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    return _encoder().encode(texts, normalize_embeddings=True)


def _faq_vectors() -> np.ndarray:
    global _faq_vecs
    if _faq_vecs is None:
        _faq_vecs = _embed([f["question"] for f in FAQ])
    return _faq_vecs


def search_faq(query: str, top_k: int = 2) -> list[dict]:
    """返回最相关的 top_k 条 FAQ，按相似度降序，附带得分。"""
    q_vec = _embed([QUERY_PREFIX + query])[0]
    sims = _faq_vectors() @ q_vec  # 两边都归一化过，点积就是余弦相似度
    order = np.argsort(-sims)[:top_k]
    return [{**FAQ[i], "score": float(sims[i])} for i in order]
