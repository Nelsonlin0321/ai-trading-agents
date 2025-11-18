from langchain.tools import tool, ToolRuntime
from src.context import Context
from src import db


@tool("Get User Investment Strategy")
async def get_user_investment_strategy(runtime: ToolRuntime[Context]):
    """
    Retrieve the current investment strategy for the trading portfolio.

    This tool fetches the current investment strategy for the trading portfolio.

    Returns
    -------
    Investment Strategy
        A string representing the current investment strategy for the trading portfolio.
    """
    try:
        await db.connect()
        bot_id = runtime.context.bot.id
        bot = await db.prisma.bot.find_unique(
            where={
                "id": bot_id
            }
        )
    except Exception as e:
        raise e
    finally:
        await db.disconnect()

    if not bot:
        raise ValueError(f"Bot with ID {bot_id} not found.")

    return bot.strategy
