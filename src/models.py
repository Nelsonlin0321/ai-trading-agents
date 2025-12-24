from pydantic import SecretStr
from langchain_deepseek import ChatDeepSeek
from src.utils import get_env

OPENROUTER_API_KEY = SecretStr(get_env("OPENROUTER_API_KEY"))
OPENROUTER_API_URL = get_env(
    "OPENROUTER_API_URL", default="https://openrouter.ai/api/v1"
)

llm_models = ["minimax/minimax-m2.1", "deepseek/deepseek-v3.2", "x-ai/grok-4.1-fast"]

reasoning_models = {"minimax/minimax-m2.1", "x-ai/grok-4.1-fast"}


def get_model(model_name: str):
    llm = ChatDeepSeek(
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_API_URL,
        model=model_name,
        extra_body={"reasoning": {"enabled": True}}
        if model_name in reasoning_models
        else None,
    )
    return llm
