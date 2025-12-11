import traceback
from loguru import logger
from prisma.types import PositionCreateInput, PositionUpdateInput, TradeCreateInput
from src.services.alpaca import get_latest_quotes
from prisma.enums import TradeType
from prisma.enums import Role
from prisma.types import RecommendCreateInput
from src import utils, db
from src.utils import async_retry
from src.tools_adaptors.base import Action
from src.tools_adaptors.utils import format_recommendations_markdown
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client


class BuyAct(Action):
    @property
    def name(self):
        return "buy_stock"

    async def arun(
        self,
        runId,
        bot_id: str,
        ticker: str,
        volume: float,
        rationale: str,
        confidence: float,
    ):
        try:
            clock = alpaca_trading_client.get_clock()
            if not clock.is_open:  # type: ignore
                return "Market is closed. Cannot buy stock."
            quotes = await get_latest_quotes([ticker])
            price = quotes["quotes"].get(ticker, {}).get("ask_price")
            if not price:
                return f"Cannot get price for {ticker}"
            price = float(price)
            total_cost = price * volume

            await db.connect()

            ticker = ticker.upper().strip()

            valid_ticker = await db.prisma.ticker.find_unique(
                where={"ticker": ticker.replace(".", "-")}
            )

            if valid_ticker is None:
                return f"Invalid ticker {ticker}"

            async with db.prisma.tx() as transaction:
                portfolio = await transaction.portfolio.find_unique(
                    where={"botId": bot_id}
                )
                if portfolio is None:
                    raise ValueError("Portfolio not found")
                if portfolio.cash < total_cost:
                    return f"Not enough cash to buy {volume} shares of {ticker} at {price} per share."
                portfolio.cash -= total_cost
                await transaction.portfolio.update(
                    where={"botId": bot_id}, data={"cash": portfolio.cash}
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
                else:
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

                    return (
                        f"Successfully bought {volume} shares of {ticker} at {price} per share. "
                        f"Current volume is {existing.volume + volume} "
                        f"with average cost {utils.format_float(existing.cost)}"
                    )

        except Exception as e:
            logger.error(f"Error buying stock: {e} Traceback: {traceback.format_exc()}")
            return f"Failed to buy {volume} shares of {ticker}"
        finally:
            await db.disconnect()


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
        try:
            clock = alpaca_trading_client.get_clock()
            if not clock.is_open:  # type: ignore
                return "Market is closed. Cannot sell stock."
            quotes = await get_latest_quotes([ticker])
            price = quotes["quotes"].get(ticker, {}).get("bid_price")
            if not price:
                return f"Cannot get price for {ticker}"
            price = float(price)
            total_proceeds = price * volume

            await db.connect()

            ticker = ticker.upper().strip()

            valid_ticker = await db.prisma.ticker.find_unique(
                where={"ticker": ticker.replace(".", "-")}
            )

            if valid_ticker is None:
                return f"Invalid ticker {ticker}"

            async with db.prisma.tx() as transaction:
                portfolio = await transaction.portfolio.find_unique(
                    where={"botId": bot_id}
                )
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

        except Exception as e:
            logger.error(
                f"Error selling stock: {e} Traceback: {traceback.format_exc()}"
            )
            return f"Failed to sell {volume} shares of {ticker}"
        finally:
            await db.disconnect()


class RecommendStockAct(Action):
    @property
    def name(self):
        return "recommend_stock"

    @utils.async_retry()
    async def arun(
        self,
        ticker: str,
        amount: float,
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
            amount: Amount of stock to buy or sell
            rationale: Rationale for the recommendation
            confidence: Confidence in the recommendation (0.0-1.0)

        Returns:
            A markdown-formatted table with the recommendation.
        """

        if not (0.0 <= confidence <= 1.0):
            return "Confidence must be between 0.0 and 1.0"

        await db.connect()
        await db.prisma.recommend.create(
            data=RecommendCreateInput(
                ticker=ticker,
                type=trade_type,
                amount=amount,
                rationale=rationale,
                confidence=confidence,
                role=role,
                runId=run_id,
                botId=bot_id,
            )
        )
        await db.disconnect()

        return (
            f"{role.value} recommended {trade_type.value} {amount} shares of {ticker}\n"
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

        await db.connect()

        recommendations = await db.prisma.recommend.find_many(
            where={
                "runId": run_id,
            },
            order={
                "role": "asc",
            },
        )

        await db.disconnect()

        return format_recommendations_markdown(recommendations)


class WriteDownTickersToReviewAct(Action):
    @property
    def name(self):
        return "write_down_tickers_to_review"

    @async_retry()
    async def arun(self, run_id: str, tickers: list[str]) -> str:
        runId = run_id
        await db.connect()

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
