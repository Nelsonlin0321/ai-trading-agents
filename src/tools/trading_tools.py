import asyncio
from loguru import logger
from typing import Literal
from prisma.enums import Role, TradeType
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime

from src.context import Context
from src.tools_adaptors.trading import (
    BuyAct,
    SellAct,
    RecommendStockAct,
    GetAnalystsRecommendationsAct,
)
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client
from src.utils.ticker import filter_valid_tickers

trading_lock = asyncio.Lock()

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
    limit_price: float | None,
    runtime: ToolRuntime[Context],
):
    """Send an order to a buy a stock.

    Args:
        ticker: Stock symbol to buy
        allocation: Additional allocation to buy (0.0-1.0). 0.05 means use 5% of the total portfolio allocation to buy.
        volume: Number of shares to buy as the reference. The actual volume to buy will be based on the allocation and portfolio value.
        rationale: Rationale for the buy order
        confidence: Confidence in the buy order (0.0-1.0)
        limit_price: Optional limit price to buy at. If None, buy at the current ask price.
        - When provided, the order will be executed only if the market price is at or below this value.
    """
    ticker = ticker.upper().strip()
    if allocation < 0.0 or allocation > 1.0:
        return "allocation must be between 0.0 and 1.0"

    invalid_tickers = await filter_valid_tickers([ticker])
    if invalid_tickers:
        return f"{', '.join(invalid_tickers)} are invalid tickers."

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    async with trading_lock:
        logger.info(
            f"Acquired lock to buy {volume} shares of {ticker} with allocation {allocation}"
        )
        result = await buy_act.arun(
            runId=runId,
            bot_id=bot_id,
            ticker=ticker,
            allocation=allocation,
            volume=volume,
            rationale=rationale,
            confidence=confidence,
            limit_price=limit_price,
        )
    logger.info(
        f"Released lock of buying {volume} shares of {ticker} with allocation {allocation}"
    )
    return result


@tool(sell_act.name)
async def sell_stock(
    ticker: str,
    allocation: float,
    volume: float,
    rationale: str,
    confidence: float,
    limit_price: float | None,
    runtime: ToolRuntime[Context],
):
    """Send an order to a sell order for a stock.

    Args:
        ticker: Stock symbol to sell
        allocation: Percentage of total portfolio(allocation) to sell (0.0-1.0). 0.05 means 5% of the total portfolio allocation.
        if it's 1.0, it means sell all the stock. if it's 0.0, it means sell nothing.
        volume: Number of shares to sell as the reference. The actual volume to sell will be based on the allocation and portfolio value.
        rationale: Rationale for the sell order
        confidence: Confidence in the sell order (0.0-1.0)
        limit_price: Optional limit price to sell at. If None, sell at the current bid price.
        - When provided, the order will be executed only if the market price is at or above this value.
    """
    ticker = ticker.upper().strip()
    if allocation < 0.0 or allocation > 1.0:
        return "allocation must be between 0.0 and 1.0"

    invalid_tickers = await filter_valid_tickers([ticker])
    if invalid_tickers:
        return f"{', '.join(invalid_tickers)} are invalid tickers."

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    async with trading_lock:
        logger.info(
            f"Acquired lock to sell {volume} shares of {ticker} with allocation {allocation}"
        )
        result = await sell_act.arun(
            runId=runId,
            bot_id=bot_id,
            ticker=ticker,
            allocation=allocation,
            volume=volume,
            rationale=rationale,
            confidence=confidence,
            limit_price=limit_price,
        )
    logger.info(
        f"Released lock of selling {volume} shares of {ticker} with allocation {allocation}"
    )
    return result


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
        """Record a BUY, SELL, or HOLD recommendation for a stock based on the current market status and its live price.

        Calling tool is mandatory to log your trading suggestions when you recommend a stock,
        capturing the ticker symbol, desired action (buy, sell, or hold),
        and allocation percentage (0.0-1.0).

        Args:
            ticker: Stock symbol to recommend to BUY, SELL, or HOLD
            allocation: Allocation of the total value of the portfolio to recommend to BUY, SELL, or HOLD: Allocation percentage (0.0-1.0) = amount / total portfolio value
            rationale: Rationale for the recommendation
            confidence: Confidence in the recommendation (0.0-1.0)
            trade_type: Whether to buy or sell the stock: `BUY`, `SELL`, or `HOLD`
        """

        bot_id = runtime.context.bot.id
        run_id = runtime.context.run.id

        invalid_tickers = await filter_valid_tickers([ticker])
        if invalid_tickers:
            return f"{', '.join(invalid_tickers)} are invalid tickers."

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
