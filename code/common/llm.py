"""模型客户端。所有期共用，读 settings 里的三个环境变量。"""

from langchain_openai import ChatOpenAI

from common import settings


def chat_model(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.MODEL_BASE_URL,
        api_key=settings.MODEL_API_KEY,
        model=settings.MODEL_NAME,
        temperature=temperature,
    )
