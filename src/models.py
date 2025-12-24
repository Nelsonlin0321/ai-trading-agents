from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from src.utils import get_env

OPENAI_API_KEY = SecretStr(get_env("OPENAI_API_KEY"))
OPENAI_API_URL = get_env("OPENAI_API_URL", default="https://api.302.ai/v1")


def get_model(model_name: str):
    if model_name.lower().startswith("gpt"):
        return ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_URL,
        )

    return ChatDeepSeek(
        model=model_name,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_URL,
    )
