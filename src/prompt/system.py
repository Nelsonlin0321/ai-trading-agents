from prisma.enums import Role

from src.context import Context
from src.prompt import SANDX_AI_INTRODUCTION, RECOMMENDATION_PROMPT, ROLE_PROMPTS_MAP
from src.tools.actions import ListPositionsAct, PortfolioPerformanceAnalysisAct


async def build_agent_system_prompt(context: Context, role: Role) -> str:
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
    sections = [
        SANDX_AI_INTRODUCTION,
        ROLE_PROMPTS_MAP[role],
        RECOMMENDATION_PROMPT,
        user_name,
        watchlist,
        positions_markdown,
        performance_narrative,
    ]
    return "\n\n".join([s for s in sections if s])
