# 可运行代码

每一期一个目录，目录名与正文章节对应。共用代码在 `common/`。

```bash
cd code
uv sync                       # 按 uv.lock 装依赖，第一次要联网
cp .env.example .env          # 填上你的模型端点
uv run python -m ep02.main    # 跑第 2 期
```

前提：一个兼容 OpenAI 协议的模型端点。本地起一个 litellm 网关最省事，见正文第 8 期；
也可以直接填任意供应商的地址和 key。Langfuse 可选，不配就不上报。
