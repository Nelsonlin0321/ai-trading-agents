from dataclasses import dataclass
from prisma.enums import Role
from prisma.models import Bot, Run
from src import db
from src.tools.actions import ListPositionsAct, PortfolioPerformanceAnalysisAct
from src.prompt import RolePrompts, SANDX_AI_INTRODUCTION


@dataclass
class UserContext:
    user_id: str


@dataclass
class Context:
    run: Run
    bot: Bot


async def get_context(run_id: str) -> Context:
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

        context = Context(run=run, bot=bot)
        await db.disconnect()
        return context
    except Exception as e:
        await db.disconnect()
        raise e


async def build_context_narrative(context: Context, role: Role) -> str:
    user = context.bot.user
    if not user:
        raise ValueError("User not found.")

    user_name = "You're serving the user: " + " ".join(
        [user.firstName or "", user.lastName or ""]
    )

    watchlist = context.bot.watchlist or []

    watchlist = (
        "Here's the watchlist of user, you can trade only on these stocks or stock in the current positions:"
        + ", ".join([w.ticker for w in watchlist])
    )

    positions_markdown = await ListPositionsAct().arun(bot_id=context.bot.id)
    performance_narrative = await PortfolioPerformanceAnalysisAct().arun(
        bot_id=context.bot.id
    )
    role_intro = RolePrompts.get(role, "")
    sections = [
        SANDX_AI_INTRODUCTION,
        role_intro,
        user_name,
        watchlist,
        positions_markdown,
        performance_narrative,
    ]
    return "\n\n".join([s for s in sections if s])
