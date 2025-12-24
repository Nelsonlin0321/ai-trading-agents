from pydantic import SecretStr
from langchain_deepseek import ChatDeepSeek
from src.utils import get_env

OPENROUTER_API_KEY = SecretStr(get_env("OPENROUTER_API_KEY"))
OPENROUTER_API_URL = get_env(
    "OPENROUTER_API_URL", default="https://openrouter.ai/api/v1"
)

THREE_TWO_ONE_API_KEY = SecretStr(get_env("THREE_TWO_ONE_API_KEY"))
THREE_TWO_ONE_API_URL = get_env(
    "THREE_TWO_ONE_API_URL", default="https://api.302.ai/v1"
)

llm_models = [
    "minimax/minimax-m2.1",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4.1-fast",
    "xiaomi/mimo-v2-flash:free",
    "openai/gpt-oss-120b:free",
    "moonshotai/kimi-k2-thinking",
]

reasoning_models = {
    "minimax/minimax-m2.1",
    "x-ai/grok-4.1-fast",
    "openai/gpt-oss-120b:free",
    "moonshotai/kimi-k2-thinking",
}

routes_to_302_ai = {
    "deepseek/deepseek-v3.2": "deepseek-v3.2",
}


def get_model(model_name: str):
    if model_name in routes_to_302_ai:
        return ChatDeepSeek(
            api_key=THREE_TWO_ONE_API_KEY,
            api_base=THREE_TWO_ONE_API_URL,
            model=routes_to_302_ai[model_name],
            # extra_body={"reasoning": {"enabled": True}},
        )

    llm = ChatDeepSeek(
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_API_URL,
        model=model_name,
        extra_body={"reasoning": {"enabled": True}}
        if model_name in reasoning_models
        else None,
    )
    return llm
