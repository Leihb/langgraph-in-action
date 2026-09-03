# 第 14 期：把第 13 期那个服务真的部署上去。
#
# 构建上下文是 code/ 这一整层（不是 ep14/ 自己）——所有期共用一份
# pyproject.toml/uv.lock，装依赖只能在那一层做：
#
#   docker build -f ep14/Dockerfile -t langgraph-ep14 .
#
# uv 装依赖的同时会把 uvx 也装进 /root/.local/bin——第 11 期起用的
# mcp-server-time 就是靠 uvx 在容器里现拉现跑，运行时镜像里必须有它，
# 不能只裁掉「用不到的开发工具」把它一起删了。
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

WORKDIR /app

# 先只拷依赖清单再 sync，改业务代码不用重新下载依赖包，Render 重新构建更快。
#
# uv.lock 里的 torch 是从 PyPI 默认索引解出来的。这个索引在 Linux 上除了
# torch 本体（aarch64 实测 454MB），还会连带装一整套 nvidia-*/cuda-*/triton
# 的 GPU 运行库（好几个 GB）。我们这个容器只用 CPU 做 embedding（第 8 期
# retrieval.py），这些全用不上。做法：sync 时把 torch 和它的 GPU 依赖整组
# 跳过，再单独从 PyTorch 官方 CPU-only 索引装 torch（同版本，159MB，没有
# GPU 依赖）。不动 pyproject.toml/uv.lock 本身——那是全书共用的，只在这个
# episode 自己的镜像构建步骤里做局部替换。
#
# 这份名单是从 uv.lock 里 grep 出来的（`grep -E '^name = "(nvidia-|triton|cuda)'`），
# 锁文件换版本后要重新对一遍。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
    --no-install-package torch \
    --no-install-package triton \
    --no-install-package cuda-bindings \
    --no-install-package cuda-pathfinder \
    --no-install-package cuda-toolkit \
    --no-install-package nvidia-cublas \
    --no-install-package nvidia-cuda-cupti \
    --no-install-package nvidia-cuda-nvrtc \
    --no-install-package nvidia-cuda-runtime \
    --no-install-package nvidia-cudnn-cu13 \
    --no-install-package nvidia-cufft \
    --no-install-package nvidia-cufile \
    --no-install-package nvidia-curand \
    --no-install-package nvidia-cusolver \
    --no-install-package nvidia-cusparse \
    --no-install-package nvidia-cusparselt-cu13 \
    --no-install-package nvidia-nccl-cu13 \
    --no-install-package nvidia-nvjitlink \
    --no-install-package nvidia-nvshmem-cu13 \
    --no-install-package nvidia-nvtx
RUN uv pip install --python .venv/bin/python torch==2.14.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY common ./common
COPY ep14 ./ep14

ENV PATH="/app/.venv/bin:$PATH"

# Render 把服务实际监听端口通过 $PORT 环境变量注入，不是固定值，
# 写死成 8000 在 Render 上会直接连不上——这是从本地 Docker 挪到 Render
# 最容易漏掉的一步。
CMD ["sh", "-c", "uvicorn ep14.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
