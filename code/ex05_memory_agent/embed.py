"""给 Store 用的 embedding 函数：跟第 8 期同一个本地模型（bge-small-zh-v1.5，512 维）。

Store 建索引要的只是一个 `list[str] -> list[list[float]]` 的函数。DeepSeek 不提供
embedding 接口，所以还是本地跑。sentence_transformers 在函数里才 import——
第 14 期量过，这个 import 一执行就是几百 MB 内存。
"""

from common import settings

DIMS = 512
_model = None


def embed(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
        except OSError:
            _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _model.encode(texts, normalize_embeddings=True).tolist()


def index_config():
    """RETRIEVAL_ENABLED=0 时不建索引：search 不带 query，按写入顺序全列出来。
    记忆只有几条到几十条时，全列出来也够用；几百条以上才必须语义检索。"""
    if not settings.RETRIEVAL_ENABLED:
        return None
    return {"embed": embed, "dims": DIMS, "fields": ["content"]}
