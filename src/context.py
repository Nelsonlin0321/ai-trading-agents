from src import db
from src.typings.context import Context


async def build_context(run_id: str):
    try:
        prisma = await db.connect()

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
        await db.disconnect()
        return context
    except Exception as e:
        await db.disconnect()
        raise e
