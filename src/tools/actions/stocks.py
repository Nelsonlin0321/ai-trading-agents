from datetime import time
from typing import TypedDict

from src.services.alpaca.api_snapshots import get_snapshots
from src.tools.actions.base import Action
from src import utils


class StockSnapshot(Action):
    @property
    def name(self):
        return "Stock Snapshot"

    async def arun(self, tickers: list[str]):
        """
        The snapshot endpoint for multiple tickers provides the latest trade, latest quote, minute bar, daily bar, and previous daily bar data for each given ticker symbol.

        Args:
            tickers: A list of ticker symbols.
        Returns:
            A dictionary of snapshot of each ticker.
        """
        data = await get_snapshots(tickers)
        return data


class PriceSnapshot(TypedDict):
    current_price: str
    current_intraday_percent: str


class StockPriceSnapshot(Action):
    @property
    def name(self):
        return "Stock Price Snapshot: Current Price And Intraday Price Change"

    async def arun(self, tickers: list[str]) -> dict[str, PriceSnapshot]:
        """
        Fetch the current price and intraday percent change for a list of tickers.

        The intraday change is calculated against the previous day's close:
        - Before 9:30 AM ET, the reference is the daily bar close.
        - At or after 9:30 AM ET, the reference is the previous daily bar close.

        Args:
            tickers: A list of ticker symbols.
        Returns:
            A dictionary mapping each ticker to its current price and intraday percent change.
        """

        ticker_price_changes = {}

        snapshots = await get_snapshots(tickers)

        for ticker in tickers:
            # Get current time in New York timezone
            current_time = utils.get_new_york_datetime().time()
            # Use dailyBar if before 9:30 AM, otherwise use prevDailyBar
            if current_time < time(9, 30):
                previous_close_price = snapshots[ticker]["dailyBar"]["c"]
            else:
                previous_close_price = snapshots[ticker]["prevDailyBar"]["c"]

            intraday_percent = (
                snapshots[ticker]["latestQuote"]["bp"] - previous_close_price) / previous_close_price

            ticker_price_changes[ticker] = PriceSnapshot(
                current_price=utils.format_currency(
                    snapshots[ticker]["latestQuote"]["bp"]),
                current_intraday_percent=utils.format_percent(
                    intraday_percent),
            )

        return ticker_price_changes
