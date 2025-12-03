# src/tools/trading_tools.py
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.tools import ToolRuntime
from src.context import Context
from src.tools_adaptors.trading import BuyAct, SellAct

buy_act = BuyAct()
sell_act = SellAct()


class BuyInput(BaseModel):
    """Input for querying latest quote for a single symbol"""

    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'")
    volume: float = Field(description="Number of shares to buy")


class SellInput(BaseModel):
    """Input for querying latest quote for a single symbol"""

    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'")
    volume: float = Field(description="Number of shares to sell")


@tool(buy_act.name, args_schema=BuyInput)
async def buy_stock(ticker: str, volume: float, runtime: ToolRuntime[Context]):
    """Execute a buy order for a stock.

    Args:
        runId: The current run ID
        bot_id: The bot ID
        ticker: Stock symbol to buy
        volume: Number of shares to buy
    """
    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await buy_act.arun(runId=runId, bot_id=bot_id, ticker=ticker, volume=volume)


@tool(sell_act.name, args_schema=SellInput)
async def sell_stock(ticker: str, volume: float, runtime: ToolRuntime[Context]):
    """Execute a sell order for a stock.

    Args:
        runId: The current run ID
        bot_id: The bot ID
        ticker: Stock symbol to sell
        volume: Number of shares to sell
    """
    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await sell_act.arun(runId=runId, bot_id=bot_id, ticker=ticker, volume=volume)
