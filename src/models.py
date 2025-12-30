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

llm_models = {
    "minimax/minimax-m2.1": {"reasoning": True, "provider": ["minimax/fp8"]},
    "deepseek/deepseek-v3.2": {"reasoning": False, "provider": None},
    "x-ai/grok-4.1-fast": {"reasoning": True, "provider": None},
    "moonshotai/kimi-k2-thinking": {"reasoning": True, "provider": None},
    "openai/gpt-oss-120b:free": {"reasoning": True, "provider": None},
    "xiaomi/mimo-v2-flash:free": {"reasoning": False, "provider": None},
}

routes_to_302_ai = {
    "deepseek/deepseek-v3.2": "deepseek-v3.2",
}


def get_model(model_name: str):
    if model_name not in llm_models:
        raise ValueError(f"Model {model_name} not found in llm_models")

    if model_name in routes_to_302_ai:
        return ChatDeepSeek(
            api_key=THREE_TWO_ONE_API_KEY,
            api_base=THREE_TWO_ONE_API_URL,
            model=routes_to_302_ai[model_name],
        )

    extra_body = get_extract_body(model_name)
    llm = ChatDeepSeek(
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_API_URL,
        model=model_name,
        extra_body=extra_body,
    )
    return llm


def get_extract_body(model_name: str):
    body = None

    if llm_models[model_name]["reasoning"]:
        body = {"reasoning": {"enabled": True}}

    if llm_models[model_name]["provider"]:
        if body:
            body["provider"] = {"order": llm_models[model_name]["provider"]}

    return body
