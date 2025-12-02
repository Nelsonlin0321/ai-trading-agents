from src import db
from src.services.sandx_ai.api_position import list_positions
from src.services.sandx_ai.api_portfolio_timeline_value import get_timeline_values


async def get_cash_balance(bot_id: str):
    try:
        await db.connect()
        portfolio = await db.prisma.portfolio.find_unique(where={"botId": bot_id})
        if portfolio is None:
            raise ValueError("Portfolio not found")
        return portfolio.cash
    except Exception as e:
        raise ValueError("Failed to get cash balance") from e
    finally:
        await db.disconnect()


__all__ = ["list_positions", "get_timeline_values"]
