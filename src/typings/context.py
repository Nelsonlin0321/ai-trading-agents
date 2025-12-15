from dataclasses import dataclass
from prisma.models import Bot, Run


@dataclass
class Context:
    run: Run
    bot: Bot
    llm_model: str
