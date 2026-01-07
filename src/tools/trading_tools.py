# src/tools/trading_tools.py
from typing import Literal
from prisma.enums import Role, TradeType
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from src.context import Context

# from src.utils.ticker import is_valid_ticker
from src.tools_adaptors.trading import (
    BuyAct,
    SellAct,
    RecommendStockAct,
    GetAnalystsRecommendationsAct,
)
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client

buy_act = BuyAct()
sell_act = SellAct()
recommend_stock_act = RecommendStockAct()
get_analysts_recommendations_act = GetAnalystsRecommendationsAct()


class BuyInput(BaseModel):
    """Input for querying latest quote for a single symbol"""

    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'")
    volume: float = Field(description="Number of shares to buy")


class SellInput(BaseModel):
    """Input for querying latest quote for a single symbol"""

    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'")
    volume: float = Field(description="Number of shares to sell")


@tool(buy_act.name)
async def buy_stock(
    ticker: str,
    allocation: float,
    volume: float,
    rationale: str,
    confidence: float,
    runtime: ToolRuntime[Context],
):
    """Send an order to a buy order for a stock.

    Args:
        ticker: Stock symbol to buy
        allocation: Additional allocation to buy (0.0-1.0). 0.05 means use 5% of the total portfolio allocation to buy.
        volume: Number of shares to buy
        rationale: Rationale for the buy order
        confidence: Confidence in the buy order (0.0-1.0)
    """
    ticker = ticker.upper().strip()
    if allocation < 0.0 or allocation > 1.0:
        return "allocation must be between 0.0 and 1.0"

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await buy_act.arun(
        runId=runId,
        bot_id=bot_id,
        ticker=ticker,
        allocation=allocation,
        volume=volume,
        rationale=rationale,
        confidence=confidence,
    )


@tool(sell_act.name)
async def sell_stock(
    ticker: str,
    allocation: float,
    volume: float,
    rationale: str,
    confidence: float,
    runtime: ToolRuntime[Context],
):
    """Send an order to a sell order for a stock.

    Args:
        ticker: Stock symbol to sell
        allocation: Percentage of total portfolio(allocation) to sell (0.0-1.0). 0.05 means 5% of the total portfolio allocation.
        if it's 1.0, it means sell all the stock. if it's 0.0, it means sell nothing.
        volume: Number of shares to sell
        rationale: Rationale for the sell order
        confidence: Confidence in the sell order (0.0-1.0)
    """
    ticker = ticker.upper().strip()
    if allocation < 0.0 or allocation > 1.0:
        return "allocation must be between 0.0 and 1.0"

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await sell_act.arun(
        runId=runId,
        bot_id=bot_id,
        ticker=ticker,
        allocation=allocation,
        volume=volume,
        rationale=rationale,
        confidence=confidence,
    )


@tool("get_market_status")
async def get_market_status():
    """Get the current market status. It's either open or closed.
    If the market is closed, you cannot buy or sell stock.
    """
    clock = alpaca_trading_client.get_clock()
    if not clock.is_open:  # type: ignore
        return "Market is closed. Cannot buy stock."

    return "Market is open. You can buy or sell stock."


def get_recommend_stock_tool(role: Role):
    @tool("recommend_stock")
    async def recommend_stock(
        ticker: str,
        allocation: float,
        rationale: str,
        confidence: float,
        trade_type: Literal["BUY", "SELL", "HOLD"],
        runtime: ToolRuntime[Context],
    ):
        """Record a BUY, SELL, or HOLD recommendation for a stock.

        Calling tool is mandatory to log your trading suggestions when you recommend a stock.
        capturing the ticker symbol, desired action (buy, sell, or hold),
        allocation percentage (0.0-1.0),

        the number of shares involved (amount = the total portfolio value * allocation, nearest integer), the reasoning behind the recommendation with the confidence level.

        These recorded recommendations can later be reviewed or aggregated to guide final investment decisions.

        Args:
            ticker: Stock symbol to recommend to BUY, SELL, or HOLD
            allocation: Allocation of the total value of the portfolio to recommend to BUY, SELL, or HOLD: Allocation percentage (0.0-1.0) = amount / total portfolio value
            rationale: Rationale for the recommendation
            confidence: Confidence in the recommendation (0.0-1.0)
            trade_type: Whether to buy or sell the stock: `BUY`, `SELL`, or `HOLD`
        """
        # ticker = ticker.upper().strip()
        # is_valid = await is_valid_ticker(ticker)
        # if not is_valid:
        #     return f"{ticker} is an invalid ticker symbol"

        bot_id = runtime.context.bot.id
        run_id = runtime.context.run.id

        return await recommend_stock_act.arun(
            run_id=run_id,
            bot_id=bot_id,
            ticker=ticker,
            allocation=allocation,
            rationale=rationale,
            confidence=confidence,
            trade_type=TradeType(trade_type),
            role=role,
        )

    return recommend_stock


@tool("get_analysts_recommendations")
async def get_analysts_recommendations(
    runtime: ToolRuntime[Context],
) -> str:
    """Retrieve consolidated analyst recommendations for final investment decisions.

    Returns:
        List of recent recommendations with keys: ticker, action (BUY/SELL/HOLD),
        price_target, rationale.
    """
    run_id = runtime.context.run.id
    return await get_analysts_recommendations_act.arun(
        run_id=run_id,
    )


@tool("get_CIO_execution_instructions")
async def get_CIO_execution_instructions(
    runtime: ToolRuntime[Context],
) -> str:
    """Retrieve consolidated CIO recommendations for the trading execution.

    Returns:
        List of CIO recommendations with keys: ticker, action (BUY/SELL/HOLD),
        price_target, rationale.
    """
    run_id = runtime.context.run.id
    return await get_analysts_recommendations_act.arun(
        run_id=run_id,
        role=Role.CHIEF_INVESTMENT_OFFICER,
    )
