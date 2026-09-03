"""环境变量读取。所有期共用，不要在各期目录里另写一份。"""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "http://localhost:4477/v1")
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "sk-local")
MODEL_NAME = os.environ.get("MODEL_NAME", "chat-default")

LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# 第 12 期起：FastAPI 服务自己的鉴权，跟 MODEL_API_KEY（打给模型端点的）
# 是两把完全不同的钥匙，别搞混——这把是"谁能调这个服务"，那把是"这个
# 服务拿什么身份去调模型"。
API_KEY = os.environ.get("API_KEY", "sk-dev-key")

# 第 13 期起：checkpointer/store 换成 Postgres 时用，不设就还用 SQLite。
POSTGRES_URL = os.environ.get("POSTGRES_URL")
