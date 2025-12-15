from src import db
from src.services.sandx_ai.api_position import list_positions
from src.services.sandx_ai.api_portfolio_timeline_value import get_timeline_values


async def get_cash_balance(bot_id: str):
    portfolio = await db.prisma.portfolio.find_unique(where={"botId": bot_id})
    if portfolio is None:
        raise ValueError("Portfolio not found")
    return portfolio.cash


__all__ = [
    "list_positions",
    "get_timeline_values",
    "get_cash_balance",
]
