from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from src.utils import get_env

OPENROUTER_API_KEY = SecretStr(get_env("OPENROUTER_API_KEY"))
OPENROUTER_API_URL = get_env(
    "OPENROUTER_API_URL", default="https://openrouter.ai/api/v1"
)


def get_model(model_name: str):
    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_URL,
        model=model_name,
    )
    return llm
