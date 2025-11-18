from langchain.tools import tool, ToolRuntime
from src.context import Context
from src import db


@tool("Get User Investment Strategy")
async def get_user_investment_strategy(runtime: ToolRuntime[Context]):
    """
    Retrieve the current investment strategy for the trading portfolio.

    This tool fetches the current investment strategy for the trading portfolio.

    Possible tool purposes:
    - Allow an AI agent to decide which assets or sectors to trade based on the user’s stated risk tolerance or philosophy.
    - Enable dynamic re-allocation logic that switches between conservative, balanced, or aggressive portfolios.
    - Provide context to downstream tools (e.g., stock screeners, rebalancers) so they filter or rank opportunities in line with the user’s mandate.
    - Surface the strategy to a dashboard or chat interface so the user can confirm or update it before orders are placed.
    - Act as a guard-rail that prevents trades violating the strategy (e.g., no crypto for a “dividend-income” strategy).

    Returns
    -------
    Investment Strategy
        A string representing the current investment strategy for the trading portfolio.
    """
    try:
        await db.connect()
        bot_id = runtime.context.bot.id
        bot = await db.prisma.bot.find_unique(where={"id": bot_id})
    except Exception as e:
        raise e
    finally:
        await db.disconnect()

    if not bot:
        raise ValueError(f"Bot with ID {bot_id} not found.")

    return bot.strategy
