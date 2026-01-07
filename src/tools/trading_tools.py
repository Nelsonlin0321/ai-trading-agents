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
    volume: float,
    rationale: str,
    confidence: float,
    runtime: ToolRuntime[Context],
):
    """Execute a buy order for a stock.

    Args:
        ticker: Stock symbol to buy
        volume: Number of shares to buy
        rationale: Rationale for the buy order
        confidence: Confidence in the buy order (0.0-1.0)
    """
    ticker = ticker.upper().strip()
    # is_valid = await is_valid_ticker(ticker)
    # if not is_valid:
    #     return f"{ticker} is an invalid ticker symbol"

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await buy_act.arun(
        runId=runId,
        bot_id=bot_id,
        ticker=ticker,
        volume=volume,
        rationale=rationale,
        confidence=confidence,
    )


@tool(sell_act.name)
async def sell_stock(
    ticker: str,
    volume: float,
    rationale: str,
    confidence: float,
    runtime: ToolRuntime[Context],
):
    """Execute a sell order for a stock.

    Args:
        ticker: Stock symbol to sell
        volume: Number of shares to sell
        rationale: Rationale for the sell order
        confidence: Confidence in the sell order (0.0-1.0)
    """
    # ticker = ticker.upper().strip()
    # is_valid = await is_valid_ticker(ticker)
    # if not is_valid:
    #     return f"{ticker} is an invalid ticker symbol"

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await sell_act.arun(
        runId=runId,
        bot_id=bot_id,
        ticker=ticker,
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
        rating: Literal["BUY", "SELL", "HOLD"],
        runtime: ToolRuntime[Context],
    ):
        """Record a allocation-based BUY, SELL, or HOLD recommendation for a stock.
        Calling tool is mandatory to log your trading suggestions when you recommend a stock.
        Args:
            ticker: Stock symbol to recommend to BUY, SELL, or HOLD
            allocation: Allocation of the total value of the portfolio to increase, decrease or hold. If the rating is hold, the allocation must be the 0.0 or same as the existing allocation.
            rationale: Detailed rationale for the recommendation based on your analysis of the stock.
            confidence: Confidence in the recommendation (0.0-1.0)
            rating: Whether to buy, sell, or hold the stock based on the allocation: `BUY`, `SELL`, or `HOLD`.
        """

        bot_id = runtime.context.bot.id
        run_id = runtime.context.run.id

        return await recommend_stock_act.arun(
            run_id=run_id,
            bot_id=bot_id,
            ticker=ticker,
            allocation=allocation,
            rationale=rationale,
            confidence=confidence,
            trade_type=TradeType(rating),
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
