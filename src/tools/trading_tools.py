# src/tools/trading_tools.py
from langchain.tools import tool
from src.tools_adaptors.trading import BuyAct, SellAct

buy_act = BuyAct()
sell_act = SellAct()


@tool(buy_act.name)
async def buy_stock(runId: str, bot_id: str, ticker: str, volume: float):
    """Execute a buy order for a stock.

    Args:
        runId: The current run ID
        bot_id: The bot ID
        ticker: Stock symbol to buy
        volume: Number of shares to buy
    """
    return await buy_act.arun(runId=runId, bot_id=bot_id, ticker=ticker, volume=volume)


@tool(sell_act.name)
async def sell_stock(runId: str, bot_id: str, ticker: str, volume: float):
    """Execute a sell order for a stock.

    Args:
        runId: The current run ID
        bot_id: The bot ID
        ticker: Stock symbol to sell
        volume: Number of shares to sell
    """
    return await sell_act.arun(runId=runId, bot_id=bot_id, ticker=ticker, volume=volume)
