from typing import Literal
from dataclasses import dataclass
from prisma.models import Bot, Run

ModelName = Literal[
    "deepseek", "gpt-5.1-thinking-plus", "deepseek-reasoner", "kimi-k2-thinking"
]


@dataclass
class Context:
    run: Run
    bot: Bot
    model_name: ModelName
