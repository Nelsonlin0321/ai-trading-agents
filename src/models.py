from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from src.typings.context import ModelName
from src.utils.config import model_mapping
from src.utils import get_env

OPENAI_API_KEY = SecretStr(get_env("OPENAI_API_KEY"))
OPENAI_API_URL = get_env("OPENAI_API_URL")


def get_model(model_name: ModelName):
    model_config_name = model_mapping[model_name]

    return ChatOpenAI(
        model=model_config_name,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_URL,
    )
