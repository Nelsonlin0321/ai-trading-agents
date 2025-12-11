from src.db import prisma
from src.typings.context import Context
from src.utils import async_retry


@async_retry(silence_error=False)
async def build_context(run_id: str):
    run = await prisma.run.find_unique(where={"id": run_id})

    if run is None:
        raise ValueError(f"Run with ID {run_id} not found.")

    if run.status != "RUNNING":
        raise ValueError(f"Run with ID {run_id} is not running.")

    bot = await prisma.bot.find_unique(
        where={"id": run.botId},
        include={
            "user": True,
            "portfolio": {"include": {"positions": True}},
            "watchlist": True,
            "trades": True,
            # "DailyPortfolioSnapshot": True,
            # "InitDailyPortfolioSnapshot": True,
            # "QQQBenchmarkPointsCache": True,
        },
    )
    if not bot:
        raise ValueError(f"Bot with ID {run.botId} not found.")

    if not bot.active:
        raise ValueError(f"Bot with ID {bot.id} is not active.")

    context = Context(run=run, bot=bot, model_name="deepseek")

    return context
