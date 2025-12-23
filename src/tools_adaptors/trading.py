from prisma.types import PositionCreateInput, PositionUpdateInput, TradeCreateInput
from src.services.alpaca import get_latest_quotes
from prisma.enums import TradeType
from prisma.enums import Role
from prisma.types import RecommendCreateInput
from datetime import datetime, timedelta, timezone
from src import utils, db
from src.utils import async_retry
from src.tools_adaptors.portfolio import calculate_latest_portfolio_value
from src.tools_adaptors.base import Action
from src.tools_adaptors.utils import format_recommendations_markdown
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client


class BuyAct(Action):
    @property
    def name(self):
        return "buy_stock"

    @async_retry()
    async def arun(
        self,
        runId,
        bot_id: str,
        ticker: str,
        volume: float,
        rationale: str,
        confidence: float,
    ):
        clock = alpaca_trading_client.get_clock()
        if not clock.is_open:  # type: ignore
            return "Market is closed. Cannot buy stock."
        quotes = await get_latest_quotes([ticker])
        price = quotes["quotes"].get(ticker, {}).get("ask_price")
        if not price:
            return f"Cannot get price for {ticker}"
        price = float(price)
        total_cost = price * volume

        ticker = ticker.upper().strip()

        valid_ticker = await db.prisma.ticker.find_unique(
            where={"ticker": ticker.replace(".", "-")}
        )

        if valid_ticker is None:
            return f"Invalid ticker {ticker}"

        async with db.prisma.tx() as transaction:
            portfolio = await transaction.portfolio.find_unique(where={"botId": bot_id})
            if portfolio is None:
                raise ValueError("Portfolio not found")
            if portfolio.cash < total_cost:
                return f"Not enough cash to buy {volume} shares of {ticker} at {price} per share."
            portfolio.cash -= total_cost
            await transaction.portfolio.update(
                where={"botId": bot_id}, data={"cash": portfolio.cash}
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
                    f"Successfully bought {volume} shares of {ticker} at {price} per share. "
                    f"Current volume is {volume} "
                    f"with average cost {utils.format_float(price)}"
                )

            await transaction.position.update(
                where={
                    "portfolioId_ticker": {
                        "portfolioId": portfolio.id,
                        "ticker": ticker,
                    }
                },
                data=PositionUpdateInput(
                    volume=existing.volume + volume,
                    cost=(existing.cost * existing.volume + price * volume)
                    / (existing.volume + volume),
                ),
            )

            return (
                f"Successfully bought {volume} shares of {ticker} at {price} per share. "
                f"Current volume is {existing.volume + volume} "
                f"with average cost {utils.format_float(existing.cost)}"
            )


class SellAct(Action):
    @property
    def name(self):
        return "sell_stock"

    @utils.async_retry()
    async def arun(
        self,
        runId,
        bot_id: str,
        ticker: str,
        volume: float,
        rationale: str,
        confidence: float,
    ):
        clock = alpaca_trading_client.get_clock()
        if not clock.is_open:  # type: ignore
            return "Market is closed. Cannot sell stock."
        quotes = await get_latest_quotes([ticker])
        price = quotes["quotes"].get(ticker, {}).get("bid_price")
        if not price:
            return f"Cannot get price for {ticker}"

        # price = float(price)
        total_proceeds = price * volume

        ticker = ticker.upper().strip()

        valid_ticker = await db.prisma.ticker.find_unique(
            where={"ticker": ticker.replace(".", "-")}
        )

        if valid_ticker is None:
            return f"Invalid ticker {ticker}"

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

            if existing.volume < volume:
                return (
                    f"Not enough shares to sell {volume} shares of {ticker}. "
                    f"Current volume is {existing.volume}."
                )

            portfolio.cash += total_proceeds
            await transaction.portfolio.update(
                where={"botId": bot_id}, data={"cash": portfolio.cash}
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
                    realizedPLPercent=(price - existing.cost) / existing.cost,
                )
            )

            if new_volume == 0:
                return (
                    f"Successfully sold {volume} shares of {ticker} at {price} per share. "
                    f"Position closed."
                )

            return (
                f"Successfully sold {volume} shares of {ticker} at {price} per share. "
                f"Current volume is {new_volume} "
                f"with average cost {utils.format_float(existing.cost)}"
            )


class RecommendStockAct(Action):
    @property
    def name(self):
        return "recommend_stock"

    @utils.async_retry()
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
            return "Confidence must be between 0.0 and 1.0"

        if not (0.0 <= allocation <= 1.0):
            return "Allocation must be between 0.0 and 1.0"

        portfolio_values = await calculate_latest_portfolio_value(bot_id)
        latest_quotes_response = await get_latest_quotes(symbols=[ticker])
        latest_quotes = latest_quotes_response["quotes"][ticker]

        price = latest_quotes["ask_price"]

        portfolio_total_value = portfolio_values["latestPortfolioValue"]
        amount = allocation * portfolio_total_value / price

        if amount < 1:
            return (
                f"Allocation is too small, the minimum amount is 1 but {amount:.2f}"
                " based on the {allocation:.1%} (allocation) * {portfolio_total_value:.2f} (total portfolio value) / {price:.2f} (price)"
            )
        amount = int(amount)
        await db.prisma.recommend.create(
            data=RecommendCreateInput(
                ticker=ticker,
                type=trade_type,
                amount=amount,
                allocation=allocation,
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

    @utils.async_retry()
    async def arun(
        self,
        run_id: str,
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
                "role": "asc",
            },
        )

        return format_recommendations_markdown(recommendations)


class WriteDownTickersToReviewAct(Action):
    @property
    def name(self):
        return "write_down_tickers_to_review"

    @async_retry()
    async def arun(self, run_id: str, tickers: list[str]) -> str:
        runId = run_id

        run = await db.prisma.run.find_unique(
            where={
                "id": runId,
            }
        )
        if not run:
            return f"Run with id {runId} not found"

        if not run.tickers:
            await db.prisma.run.update(
                where={
                    "id": runId,
                },
                data={
                    "tickers": ",".join(tickers),
                },
            )
            return f"Tickers {tickers} written down to review"
        else:
            existing_tickers = set([t.strip().upper() for t in run.tickers.split(",")])
            new_tickers = set([t.strip().upper() for t in tickers])

            if existing_tickers != new_tickers:
                return (
                    "The tickers to review are not the same as the ones that user specified."
                    "Please check the tickers and try again. "
                    "The tickers that user specified are: "
                    f"{', '.join(new_tickers)}"
                    "The tickers that you are going to write down are: "
                    f"{', '.join(new_tickers)}"
                )
            else:
                return f"Tickers {tickers} written down to review"


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

    asyncio.run(test_TradeHistoryAct())
