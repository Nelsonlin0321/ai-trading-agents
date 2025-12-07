from dataclasses import dataclass
from prisma.models import Bot, Run
from src.typings.models import ModelName


@dataclass
class Context:
    run: Run
    bot: Bot
    model_name: ModelName
