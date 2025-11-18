from langchain.agents.middleware import SummarizationMiddleware, TodoListMiddleware
from src.models import get_model

langchain_model = get_model("deepseek")

summarization_middleware = SummarizationMiddleware(
    model=langchain_model,
    max_tokens_before_summary=128_000,
    messages_to_keep=20,
)


todo_list_middleware = TodoListMiddleware()

__all__ = [
    "summarization_middleware",
    "todo_list_middleware",
]
