from typing import Optional
from prisma.types import PositionCreateInput, PositionUpdateInput, TradeCreateInput
from src.services.alpaca import get_latest_quotes
from prisma.enums import TradeType
from prisma.enums import Role
from prisma.types import RecommendCreateInput
from prisma.models import Recommend
from datetime import datetime, timedelta, timezone
from src import utils, db
from src.utils import async_retry
from src.tools_adaptors.portfolio import (
    calculate_latest_portfolio_value,
)
from src.tools_adaptors.base import Action
from src.tools_adaptors.utils import format_recommendations_markdown
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client


class BuyAct(Action):
    @property
    def name(self):
        return "buy_stock"

    # @async_retry()
    async def arun(
        self,
        runId,
        bot_id: str,
        ticker: str,
        volume: float,
        allocation: float,  # additional allocation to buy
        rationale: str,
        confidence: float,
        limit_price: Optional[float] = None,
    ):
        clock = alpaca_trading_client.get_clock()
        if not clock.is_open:  # type: ignore
            return "Market is closed. Cannot buy stock."
        ticker = ticker.upper().strip()
        quotes = await get_latest_quotes([ticker])
        price = quotes["quotes"].get(ticker, {}).get("ask_price")
        if not price:
            return f"Cannot get price for {ticker}"

        if limit_price:
            if price > limit_price:
                return (
                    f"Current Price ${price} is higher than limit price (Target price to buy at) ${limit_price}. Cannot buy."
                    "Confirm with the Chief Investment Officer if he wants to buy at the market price."
                )

        price = float(price)
        generated_volume = volume
        generated_allocation = allocation

        portfolio_values_response = await calculate_latest_portfolio_value(bot_id)
        total_portfolio_value = portfolio_values_response["latestPortfolioValue"]

        allocated_value_to_increase = total_portfolio_value * generated_allocation
        volume = int(allocated_value_to_increase / float(price))
        total_cost = price * volume

        allocation = total_cost / total_portfolio_value

        if volume == 0:
            return (
                f"Allocation :{utils.format_percent(generated_allocation)} {utils.format_float(allocated_value_to_increase)}  "
                f"is not enough to buy at least 1 stock with the current price:{utils.format_float(price)}. "
                "The price is higher than the allocation value."
            )

        async with db.prisma.tx() as transaction:
            portfolio = await transaction.portfolio.find_unique(where={"botId": bot_id})
            if portfolio is None:
                raise ValueError("Portfolio not found")
            if portfolio.cash < total_cost:
                return f"Not enough cash to buy {volume} shares of {ticker} at {price} per share."

            await transaction.portfolio.update(
                where={"botId": bot_id}, data={"cash": {"decrement": total_cost}}
            )

            await transaction.trade.create(
                data=TradeCreateInput(
                    rationale=rationale,
                    confidence=confidence,
                    type=TradeType.BUY,
                    price=price,
                    ticker=ticker,
                    amount=volume,
                    runId=runId,
                    botId=bot_id,
                )
            )

            existing = await transaction.position.find_unique(
                where={
                    "portfolioId_ticker": {
                        "portfolioId": portfolio.id,
                        "ticker": ticker,
                    }
                }
            )

            if existing is None:
                await transaction.position.create(
                    data=PositionCreateInput(
                        ticker=ticker,
                        volume=volume,
                        portfolioId=portfolio.id,
                        cost=price,
                    )
                )

                return (
                    f"Order received to increase allocation by {utils.format_percent(generated_allocation)} through the purchase of {utils.format_float(generated_volume)} shares of {ticker}.\n\n"
                    f"Post-trade result:\n\n"
                    f"Allocation successfully increased from 0% to {utils.format_percent(allocation)} "
                    f"via the purchase of {volume} shares of {ticker} at {price} per share.\n\n"
                    "Note: The executed order may differ from the original instruction due to share-volume rounding or misalignment between the requested allocation and share quantity. "
                    "Please regard the post-trade allocation and executed volume as final."
                )

            new_volume = existing.volume + volume
            await transaction.position.update(
                where={
                    "portfolioId_ticker": {
                        "portfolioId": portfolio.id,
                        "ticker": ticker,
                    }
                },
                data=PositionUpdateInput(
                    volume=new_volume,
                    cost=(existing.cost * existing.volume + price * volume)
                    / (new_volume),
                ),
            )

            new_allocation = new_volume * price / total_portfolio_value

            return (
                f"Order received to increase allocation by {utils.format_percent(generated_allocation)} through the purchase of {utils.format_float(generated_volume)} shares of {ticker}.\n\n"
                f"Post-trade result:\n\n"
                f"Allocation successfully increased by {utils.format_percent(allocation)} to {utils.format_percent(new_allocation)} "
                f"via the purchase of {volume} shares of {ticker} at {price} per share.\n\n"
                "Note: The executed order may differ from the original instruction due to share-volume rounding or misalignment between the requested allocation and share quantity. "
                "Please regard the post-trade allocation and executed volume as final."
            )


class SellAct(Action):
    @property
    def name(self):
        return "sell_stock"

    @async_retry()
    async def arun(
        self,
        runId,
        bot_id: str,
        allocation: float,
        ticker: str,
        volume: float,
        rationale: str,
        confidence: float,
        limit_price: Optional[float] = None,
    ):
        clock = alpaca_trading_client.get_clock()
        if not clock.is_open:  # type: ignore
            return "Market is closed. Cannot sell stock."

        ticker = ticker.upper().strip()
        quotes = await get_latest_quotes([ticker])
        price = quotes["quotes"].get(ticker, {}).get("bid_price")
        if not price:
            return f"Cannot get price for {ticker}"

        if limit_price:
            if price < limit_price:
                return (
                    f"Current Price ${price} is lower than limit price (Target price to sell at) ${limit_price}. Cannot sell."
                    "Confirm with the Chief Investment Officer if he wants to sell at the market price."
                )

        price = float(price)
        generated_volume = volume
        generated_allocation = allocation

        portfolio_values_response = await calculate_latest_portfolio_value(bot_id)
        total_portfolio_value = portfolio_values_response["latestPortfolioValue"]

        allocated_value_to_decrease = total_portfolio_value * generated_allocation
        volume = int(allocated_value_to_decrease / price)
        total_proceeds = price * volume
        allocation = total_proceeds / total_portfolio_value

        if volume == 0:
            return (
                f"Allocation :{utils.format_percent(generated_allocation)} {utils.format_float(allocated_value_to_decrease)}  "
                f"is not enough to sell at least 1 stock with the current price:{utils.format_float(price)}."
                "The price is higher than the allocation value."
            )

        async with db.prisma.tx() as transaction:
            portfolio = await transaction.portfolio.find_unique(where={"botId": bot_id})
            if portfolio is None:
                raise ValueError("Portfolio not found")

            existing = await transaction.position.find_unique(
                where={
                    "portfolioId_ticker": {
                        "portfolioId": portfolio.id,
                        "ticker": ticker,
                    }
                }
            )

            if existing is None:
                return f"No position found for {ticker}"

            if existing.volume == 0:
                return f"Cannot sell {ticker} because the current volume is 0."

            if existing.volume < volume:
                return (
                    f"Not enough shares to sell {volume} shares of {ticker}. "
                    f"Current volume is {existing.volume}."
                )

            await transaction.portfolio.update(
                where={"botId": bot_id}, data={"cash": {"increment": total_proceeds}}
            )

            new_volume = existing.volume - volume

            if new_volume == 0:
                await transaction.position.delete(
                    where={
                        "portfolioId_ticker": {
                            "portfolioId": portfolio.id,
                            "ticker": ticker,
                        }
                    }
                )
            else:
                await transaction.position.update(
                    where={
                        "portfolioId_ticker": {
                            "portfolioId": portfolio.id,
                            "ticker": ticker,
                        }
                    },
                    data=PositionUpdateInput(
                        volume=new_volume,
                    ),
                )

            await transaction.trade.create(
                data=TradeCreateInput(
                    rationale=rationale,
                    confidence=confidence,
                    type=TradeType.SELL,
                    price=price,
                    ticker=ticker,
                    amount=volume,
                    runId=runId,
                    botId=bot_id,
                    realizedPL=price * volume - existing.cost * volume,
                    realizedPLPercent=((price - existing.cost) / existing.cost)
                    if existing.cost != 0
                    else 0,
                )
            )

            if new_volume == 0:
                return (
                    f"Order received to decrease allocation by {utils.format_percent(generated_allocation)} through the purchase of {utils.format_float(generated_volume)} shares of {ticker}.\n\n"
                    f"Post-trade result:\n\n"
                    f"Allocation successfully decreased from {utils.format_percent(allocation)} to 0%, closing the position, "
                    f"via the selling of {volume} shares of {ticker} at {price} per share.\n\n"
                    "Note: The executed order may differ from the original instruction due to share-volume rounding or misalignment between the requested allocation and share quantity. "
                    "Please regard the post-trade allocation and executed volume as final."
                )

            new_allocation = new_volume * price / total_portfolio_value

            return (
                f"Order received to decrease allocation by {utils.format_percent(generated_allocation)} through the purchase of {utils.format_float(generated_volume)} shares of {ticker}.\n\n"
                f"Post-trade result:\n\n"
                f"Allocation successfully decreased by {utils.format_percent(allocation)} to {utils.format_percent(new_allocation)}, "
                f"via the selling of {volume} shares of {ticker} at {price} per share.\n\n"
                "Note: The executed order may differ from the original instruction due to share-volume rounding or misalignment between the requested allocation and share quantity. "
                "Please regard the post-trade allocation and executed volume as final."
            )


class RecommendStockAct(Action):
    @property
    def name(self):
        return "recommend_stock"

    @async_retry()
    async def arun(
        self,
        ticker: str,
        allocation: float,
        rationale: str,
        confidence: float,
        trade_type: TradeType,
        role: Role,
        run_id: str,
        bot_id: str,
    ) -> str:
        """
        Recommend a stock to buy or sell.

        Args:
            ticker: Stock ticker, e.g. 'AAPL'
            allocation: Allocation of the portfolio to the stock (0.0-1.0)
            rationale: Rationale for the recommendation
            confidence: Confidence in the recommendation (0.0-1.0)

        Returns:
            A markdown-formatted table with the recommendation.
        """

        if not (0.0 <= confidence <= 1.0):
            return f"{ticker} Confidence must be between 0.0 and 1.0"

        if not (0.0 <= allocation <= 1.0):
            return f"{ticker} Allocation must be between 0.0 and 1.0"

        portfolio_values = await calculate_latest_portfolio_value(bot_id)
        latest_quotes_response = await get_latest_quotes(symbols=[ticker])
        latest_quotes = latest_quotes_response["quotes"][ticker]

        price = latest_quotes["ask_price"]

        portfolio_total_value = portfolio_values["latestPortfolioValue"]
        amount = allocation * portfolio_total_value / price

        if amount < 1:
            return (
                f"{ticker} Allocation is too small, the minimum amount is 1 but {amount:.2f}"
                f" based on the {allocation:.1%} (allocation) * {portfolio_total_value:.2f} (total portfolio value) / {price:.2f} (price). "
                "Please adjust the allocation percentage to at least 1 share."
            )
        amount = int(amount)
        await db.prisma.recommend.create(
            data=RecommendCreateInput(
                ticker=ticker,
                type=trade_type,
                amount=amount,
                allocation=allocation,
                limitPrice=latest_quotes["ask_price"]
                if trade_type == TradeType.BUY
                else latest_quotes["bid_price"],
                rationale=rationale,
                confidence=confidence,
                role=role,
                runId=run_id,
                botId=bot_id,
            )
        )

        return (
            f"{role.value} recommended {trade_type.value} {amount} shares of {ticker} "
            f"with allocation {allocation:.1%} of {portfolio_total_value:.2f} (total portfolio value)\n"
            f"Confidence: {confidence:.1%}\n"
            f"Rationale: {rationale}"
        )


class GetAnalystsRecommendationsAct(Action):
    @property
    def name(self):
        return "get_analysts_recommendations"

    @async_retry()
    async def arun(
        self,
        run_id: str,
        role: Optional[Role] = None,
    ) -> str:
        """
        Get analysts recommendations.

        Args:
            run_id: Run ID
            bot_id: Bot ID

        Returns:
            A markdown-formatted table with the recommendations.
        """

        recommendations = await db.prisma.recommend.find_many(
            where={
                "runId": run_id,
            },
            order={
                "createdAt": "desc",
            },
        )

        grouped_recs: dict[str, Recommend] = {}
        for rec in recommendations:
            key = f"{rec.role}-{rec.ticker}"
            if key not in grouped_recs:
                if role is not None:
                    if rec.role == role.value:
                        grouped_recs[key] = rec
                else:
                    grouped_recs[key] = rec

        recommendations = sorted(list(grouped_recs.values()), key=lambda x: str(x.role))

        return format_recommendations_markdown(recommendations)


class TradeHistoryAct(Action):
    @property
    def name(self):
        return "get_trade_history_in_30_days"

    @async_retry()
    async def arun(self, run_id: str) -> str:
        run = await db.prisma.run.find_unique(where={"id": run_id})
        if not run:
            return f"Run with id {run_id} not found"

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

        trades = await db.prisma.trade.find_many(
            where={
                "botId": run.botId,
                "createdAt": {"gte": cutoff_date},
            },
            order={"createdAt": "desc"},
        )

        if not trades:
            return "No trades found in the last 30 days."

        tickers = list(set([t.ticker for t in trades]))
        quotes_response = await get_latest_quotes(tickers)
        quotes = quotes_response.get("quotes", {})

        lines = [
            "Notes:",
            "- `Date`: trade timestamp (UTC, YYYY-MM-DD HH:MM)",
            "- `Ticker`: instrument symbol",
            "- `Type`: `BUY` or `SELL`",
            "- `Amount`: shares; negative for `SELL`, positive for `BUY`",
            "- `Trade Price`: executed price at trade time",
            "- `Current Price`: latest bid price used in PnL calculations",
            "- `Realized PnL` (SELL): (sell - cost) × amount; `%` = (sell - cost) / cost",
            "- `Unrealized PnL` (BUY): (bid - trade price) × amount; `%` = (bid - trade price) / trade price",
            "- `PnL if Held` (SELL): (bid - sell price) × amount; `%` = (bid - sell price) / sell price",
            "- `PnL if Held` interpretation: > 0 holding would outperform selling; < 0 selling was favorable; ≈ 0 neutral impact",
            "- `PnL if Held` purpose: quick post-trade indicator to evaluate sell timing and potential profit/loss if held; not for a re-entry recommendation",
            "- `Rationale`: brief explanation of the decision",
            "",
            "| Date | Ticker | Type | Amount | Trade Price | Current Price | Realized PnL | Realized PnL % | Unrealized PnL | Unrealized PnL % | PnL if Held | PnL % if Held | Rationale |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for trade in trades:
            date_str = trade.createdAt.strftime("%Y-%m-%d %H:%M")

            amount = trade.amount
            realized_pnl_str = "N/A"
            realized_pnl_percent_str = "N/A"
            unrealized_pnl_str = "N/A"
            unrealized_pnl_percent_str = "N/A"
            pnl_if_held_str = "N/A"
            pnl_percent_if_held_str = "N/A"
            current_price_str = "N/A"
            quote = quotes.get(trade.ticker)
            bid_price: float | None = None
            if quote and quote.get("bid_price"):
                bid_price = float(quote["bid_price"])
                current_price_str = utils.format_float(bid_price)

            if trade.type == TradeType.SELL:
                amount = -1 * amount
                if trade.realizedPL is not None:
                    realized_pnl_str = utils.format_float(trade.realizedPL)
                if trade.realizedPLPercent is not None:
                    realized_pnl_percent_str = utils.format_percent(
                        trade.realizedPLPercent
                    )
                if bid_price is not None:
                    pnl_if_held = (bid_price - trade.price) * trade.amount
                    pnl_percent_if_held = (bid_price - trade.price) / trade.price
                    pnl_if_held_str = utils.format_float(pnl_if_held)
                    pnl_percent_if_held_str = utils.format_percent(pnl_percent_if_held)

            elif trade.type == TradeType.BUY:
                if bid_price is not None:
                    pnl = (bid_price - trade.price) * trade.amount
                    pnl_percent = (bid_price - trade.price) / trade.price
                    unrealized_pnl_str = utils.format_float(pnl)
                    unrealized_pnl_percent_str = utils.format_percent(pnl_percent)

            lines.append(
                "| "
                + " | ".join(
                    [
                        date_str,
                        trade.ticker,
                        trade.type,
                        str(amount),
                        str(trade.price),
                        current_price_str,
                        realized_pnl_str,
                        realized_pnl_percent_str,
                        unrealized_pnl_str,
                        unrealized_pnl_percent_str,
                        pnl_if_held_str,
                        pnl_percent_if_held_str,
                        trade.rationale,
                    ]
                )
                + " |"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    #  python -m src.tools_adaptors.trading
    import asyncio
    from src import db

    async def test_TradeHistoryAct():
        await db.connect()
        act = TradeHistoryAct()
        run_id = "aeb9c6eb-7a16-442c-a2ed-93f92788fccd"
        result = await act.arun(run_id)
        print(result)
        await db.disconnect()

    async def test_get_analyst_recommendation():
        await db.connect()
        act = GetAnalystsRecommendationsAct()
        run_id = "143d857a-f180-4435-b38b-89a4cb4b8e84"
        result = await act.arun(run_id, role=Role("CHIEF_INVESTMENT_OFFICER"))
        print(result)
        await db.disconnect()

    asyncio.run(test_TradeHistoryAct())
    asyncio.run(test_get_analyst_recommendation())
