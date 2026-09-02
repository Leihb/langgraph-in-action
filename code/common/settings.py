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
