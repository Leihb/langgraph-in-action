"""环境自检：模型端点通不通、版本对不对。第 1 期"跑起来"用。

    uv run python -m common.check
"""

import sys
from importlib.metadata import version

from langchain_openai import ChatOpenAI

from common import settings

PINNED = {"langgraph": "1.2.11", "langchain": "1.3.18"}


def main() -> int:
    print(f"python      {sys.version.split()[0]}")
    ok = True
    for name, want in PINNED.items():
        got = version(name)
        mark = "ok" if got == want else f"expected {want}"
        ok &= got == want
        print(f"{name:<12}{got:<10}{mark}")

    print(f"endpoint    {settings.MODEL_BASE_URL}")
    print(f"model       {settings.MODEL_NAME}")
    print(f"langfuse    {'on' if settings.LANGFUSE_ENABLED else 'off (未配置，不上报)'}")

    llm = ChatOpenAI(
        base_url=settings.MODEL_BASE_URL,
        api_key=settings.MODEL_API_KEY,
        model=settings.MODEL_NAME,
        temperature=0,
        max_tokens=20,
    )
    try:
        reply = llm.invoke("只回复两个字：收到")
    except Exception as e:  # noqa: BLE001 —— 自检脚本，任何异常都直接给读者看
        print(f"model call  failed: {type(e).__name__}: {e}")
        return 1
    print(f"model call  ok -> {reply.content.strip()!r}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
