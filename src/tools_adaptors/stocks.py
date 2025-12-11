import asyncio
from datetime import time, datetime, timedelta, timezone, date
from typing import TypedDict, List
from src.services.alpaca import (
    get_snapshots,
    get_historical_price_bars,
    get_most_active_stocks,
    get_latest_quotes,
)
from src.services.alpaca.typing import PriceBar
from src.tools_adaptors.base import Action
from src import utils
from src.utils.constants import ETF_TICKERS
from src.utils import async_retry


class StockRawSnapshotAct(Action):
    @property
    def name(self):
        return "Get Stock Snapshot"

    @async_retry()
    async def arun(self, tickers: list[str]) -> dict:
        """
        Fetch raw market snapshots for multiple tickers.

        Returns the latest trade, latest quote, minute bar, daily bar, and previous daily bar data
        for each ticker symbol.

        Args:
            tickers: A list of ticker symbols.
        Returns:
            A dictionary mapping each ticker to its complete raw snapshot data.
        """
        data = await get_snapshots(tickers)
        return data


class CurrentPriceAndIntradayChange(TypedDict):
    current_price: str
    current_intraday_percent: str


class StockCurrentPriceAndIntradayChangeAct(Action):
    @property
    def name(self):
        return "Stock Current Price and Intraday Change"

    @async_retry()
    async def arun(
        self, tickers: list[str]
    ) -> dict[str, CurrentPriceAndIntradayChange]:
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
                snapshots[ticker]["latestQuote"]["bp"] - previous_close_price
            ) / previous_close_price

            ticker_price_changes[ticker] = CurrentPriceAndIntradayChange(
                current_price=utils.format_currency(
                    snapshots[ticker]["latestQuote"]["bp"]
                ),
                current_intraday_percent=utils.format_percent(intraday_percent),
            )

        return ticker_price_changes


class HistoricalPriceChangePeriods(TypedDict):
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class StockHistoricalPriceChangesAct(Action):
    @property
    def name(self):
        return "Stock Historical Price Changes"

    # disable: pylint:disable=too-many-locals
    @async_retry()
    async def arun(self, tickers: list[str]) -> dict[str, HistoricalPriceChangePeriods]:
        """
        Compute percentage changes over standard periods using Alpaca daily bars.

        Periods: one_day, one_week, one_month, three_months, six_months, one_year, three_years.

        - Uses UTC timestamps (ISO 8601 with trailing 'Z') as required by Alpaca.
        - Selects the most recent close as the "current" reference.
        - For each period, finds the close on or before the target date.
        - Returns None when not enough history exists for a period.

        Args:
            tickers: List of ticker symbols.
        Returns:
            Mapping of ticker to period percent changes, e.g. {"AAPL": {"1w": 0.02, ...}}.
        """

        def _to_iso_z(dt: datetime) -> str:
            return (
                dt.astimezone(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )

        def _parse_ts(ts: str) -> datetime:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )

        def _find_close_on_or_before(
            bars: List[PriceBar], target_dt: datetime
        ) -> float | None:
            for bar_ in bars:
                bar_dt = _parse_ts(bar_["timestamp"])
                if bar_dt <= target_dt:
                    return bar_["close_price"]
            return None

        period_deltas: dict[str, timedelta] = {
            "one_day": timedelta(days=1),
            "one_week": timedelta(days=7),
            "one_month": timedelta(days=30),
            "three_months": timedelta(days=90),
            "six_months": timedelta(days=180),
            # "one_year": timedelta(days=365),
            # "three_years": timedelta(days=365 * 3),
        }

        start = (date.today() - timedelta(days=180 + 7)).isoformat()
        end = date.today().isoformat()

        bars_by_symbol = await get_historical_price_bars(
            symbols=tickers,
            timeframe="1Day",
            start=start,
            end=end,
            sort="desc",  # IMPORTANT: sort by descending timestamp
        )

        results = {}
        for ticker in tickers:
            if not bars_by_symbol.get(ticker):
                continue
            bars = bars_by_symbol[ticker]
            latest_close = float(bars[0]["close_price"])
            latest_dt = _parse_ts(bars[0]["timestamp"])

            period_changes: dict[str, str | None] = {
                "one_day": None,
                "one_week": None,
                "one_month": None,
                "three_months": None,
                "six_months": None,
                "one_year": None,
                "three_years": None,
            }

            for key, delta in period_deltas.items():
                target_dt = latest_dt - delta
                prior_close = _find_close_on_or_before(bars, target_dt)
                if prior_close is not None and prior_close != 0:
                    period_changes[key] = utils.format_percent_change(
                        (latest_close - prior_close) / prior_close
                    )
                else:
                    period_changes[key] = None

            results[ticker] = period_changes

        return results


class StockPriceSnapshotWithHistory(TypedDict):
    current_price: str
    current_intraday_percent: str
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class ETFPriceSnapshotWithHistory(TypedDict):
    ticker: str
    name: str
    description: str
    current_price: str
    current_intraday_percent: str
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class StockLivePriceChangeAct(Action):
    @property
    def name(self):
        return "get_stock_live_price_and_change"

    # To need to retry because the sub-actions has retry decorator
    # @async_retry()
    async def arun(
        self, tickers: list[str]
    ) -> dict[str, StockPriceSnapshotWithHistory]:
        """
        Fetch a complete price snapshot for multiple tickers.

        Returns current price, intraday change, and historical percent changes
        for standard periods (1D, 1W, 1M, 3M, 6M, 1Y, 3Y) in a single call.
        """

        current_task = asyncio.create_task(
            StockCurrentPriceAndIntradayChangeAct().arun(tickers)
        )
        historical_task = asyncio.create_task(
            StockHistoricalPriceChangesAct().arun(tickers)
        )

        current, historical = await asyncio.gather(current_task, historical_task)

        results = {}
        if historical == "ERROR":
            return results

        if current == "ERROR":
            return results

        for ticker in tickers:
            history: HistoricalPriceChangePeriods = historical[ticker]
            currency: CurrentPriceAndIntradayChange = current[ticker]
            results[ticker] = StockPriceSnapshotWithHistory(
                current_price=currency["current_price"],
                current_intraday_percent=currency["current_intraday_percent"],
                one_day=history["one_day"],
                one_week=history["one_week"],
                one_month=history["one_month"],
                three_months=history["three_months"],
                six_months=history["six_months"],
                one_year=history["one_year"],
                three_years=history["three_years"],
            )

        return results


class ETFLivePriceChangeAct(Action):
    @property
    def name(self):
        return "get_major_etf_live_price_and_historical_change"

    # No need to retry because the sub-actions has retry decorator
    # @async_retry()
    async def arun(self) -> dict[str, ETFPriceSnapshotWithHistory]:
        """
        Fetch a complete price snapshot for multiple major ETF tickers.

        Returns current price, intraday change, and historical percent changes
        for standard periods (1D, 1W, 1M, 3M, 6M, 1Y, 3Y) in a single call.
        """
        tickers = [t["ticker"] for t in ETF_TICKERS]
        ticker_info_dict = {t["ticker"]: t for t in ETF_TICKERS}

        current_task = asyncio.create_task(
            StockCurrentPriceAndIntradayChangeAct().arun(tickers)
        )
        historical_task = asyncio.create_task(
            StockHistoricalPriceChangesAct().arun(tickers)
        )

        current, historical = await asyncio.gather(current_task, historical_task)

        results = {}
        if current == "ERROR":
            return results
        if historical == "ERROR":
            return results

        for ticker in tickers:
            history: HistoricalPriceChangePeriods = historical[ticker]
            currency: CurrentPriceAndIntradayChange = current[ticker]
            results[ticker] = ETFPriceSnapshotWithHistory(
                ticker=ticker,
                name=ticker_info_dict[ticker]["name"],
                description=ticker_info_dict[ticker]["description"],
                current_price=currency["current_price"],
                current_intraday_percent=currency["current_intraday_percent"],
                one_day=history["one_day"],
                one_week=history["one_week"],
                one_month=history["one_month"],
                three_months=history["three_months"],
                six_months=history["six_months"],
                one_year=history["one_year"],
                three_years=history["three_years"],
            )

        return results


class ActiveStockFullPriceMetrics(TypedDict):
    symbol: str
    trade_count: int
    volume: int
    current_price: str
    current_intraday_percent: str
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class MostActiveStockFullPriceMetrics(TypedDict):
    last_updated: str
    most_actives: list[ActiveStockFullPriceMetrics]


class MostActiveStockersAct(Action):
    @property
    def name(self):
        return "get_most_active_stockers_with_historical_price_changes"

    @async_retry()
    async def arun(self):
        """
        Get the most active stockers.
        """
        data = await get_most_active_stocks()
        tickers = [item["symbol"] for item in data["most_actives"]]
        most_active_stocks: list[ActiveStockFullPriceMetrics] = []

        results = await StockLivePriceChangeAct().arun(tickers)

        for item in data["most_actives"]:
            symbol = item["symbol"]
            price_metrics = results.get(symbol)
            if price_metrics:
                full_metrics = ActiveStockFullPriceMetrics(
                    symbol=symbol,
                    trade_count=item["trade_count"],
                    volume=item["volume"],
                    **price_metrics,
                )
                most_active_stocks.append(full_metrics)

        return MostActiveStockFullPriceMetrics(
            last_updated=data["last_updated"],
            most_actives=most_active_stocks,
        )


# src/tools_adaptors/stocks.py - Add this class
class MultiLatestQuotesAct(Action):
    @property
    def name(self):
        return "get_multi_symbols_latest_quotes"

    @async_retry()
    async def arun(self, symbols: list[str]) -> str:
        """
        Fetch latest quotes for multiple symbols.

        Returns:
            dict: Mapping of symbol to quote data including:
                - ask_price: Current ask price
                - ask_size: Ask size
                - bid_price: Current bid price
                - bid_size: Bid size
                - timestamp: Quote timestamp
        """

        quotesResponse = await get_latest_quotes(symbols)
        formatted_quotes = []
        quotes = quotesResponse["quotes"]
        for symbol in quotes:
            quote = quotes[symbol]
            formatted_quote = {
                "symbol": symbol,
                "bid_price": utils.format_currency(quote["bid_price"]),
                "bid_size": utils.human_format(quote["bid_size"]),
                "ask_price": utils.format_currency(quote["ask_price"]),
                "ask_size": utils.human_format(quote["ask_size"]),
                "spread": utils.format_currency(
                    quote["ask_price"] - quote["bid_price"]
                ),
                "spread_percent": utils.format_percent(
                    (quote["ask_price"] - quote["bid_price"]) / quote["bid_price"]
                ),
                "exchange": f"{quote['bid_exchange']}/{quote['ask_exchange']}",
                "timestamp": utils.format_datetime(quote["timestamp"]),
                # "conditions": ", ".join(quote["conditions"]) if quote["conditions"] else "Normal"
            }
            formatted_quotes.append(formatted_quote)

        if not formatted_quotes:
            return "No quote data available for the requested symbols."

        markdown_table = utils.dicts_to_markdown_table(formatted_quotes)
        heading = "## Latest Market Quotes"
        note = f"""
**Note:**
- Data fetched at {utils.get_current_timestamp()} New York time
- Bid/Ask prices are real-time consolidated quotes
- Spread = Ask Price - Bid Price
- Conditions: 'R' = Regular Market, 'O' = Opening Quote, 'C' = Closing Quote
"""

        return heading + "\n\n" + note + "\n\n" + markdown_table


class SingleLatestQuotesAct(Action):
    @property
    def name(self):
        return "get_single_symbol_latest_quotes"

    @async_retry()
    async def arun(self, symbol: str) -> str:
        """
        Fetch latest quotes for a single symbol.

        Returns:
            dict: Mapping of symbol to quote data including:
                - ask_price: Current ask price
                - ask_size: Ask size
                - bid_price: Current bid price
                - bid_size: Bid size
                - timestamp: Quote timestamp
        """

        quotes = await get_latest_quotes([symbol])
        quote = quotes.get(symbol)

        if not quote:
            return "No quote data available for the requested symbol."

        formatted_quote = {
            "symbol": symbol,
            "bid_price": utils.format_currency(quote["bid_price"]),
            "bid_size": utils.human_format(quote["bid_size"]),
            "ask_price": utils.format_currency(quote["ask_price"]),
            "ask_size": utils.human_format(quote["ask_size"]),
            "spread": utils.format_currency(quote["ask_price"] - quote["bid_price"]),
            "spread_percent": utils.format_percent(
                (quote["ask_price"] - quote["bid_price"]) / quote["bid_price"]
            ),
            "exchange": f"{quote['bid_exchange']}/{quote['ask_exchange']}",
            "timestamp": utils.format_datetime(quote["timestamp"]),
            # "conditions": ", ".join(quote["conditions"]) if quote["conditions"] else "Normal"
        }

        markdown_table = utils.dict_to_markdown_table(formatted_quote)
        heading = f"## Latest {symbol} Market Quotes"
        note = f"""
**Note:**
- Data fetched at {utils.get_current_timestamp()} New York time
- Bid/Ask prices are real-time consolidated quotes
- Spread = Ask Price - Bid Price
- Conditions: 'R' = Regular Market, 'O' = Opening Quote, 'C' = Closing Quote
"""

        return heading + "\n\n" + note + "\n\n" + markdown_table


# only Act
__all__ = [
    "StockRawSnapshotAct",
    "StockCurrentPriceAndIntradayChangeAct",
    "StockHistoricalPriceChangesAct",
    "StockLivePriceChangeAct",
    "ETFLivePriceChangeAct",
    "MostActiveStockersAct",
    "SingleLatestQuotesAct",
    "MultiLatestQuotesAct",
]

if __name__ == "__main__":
    # python -m src.tools.actions.stocks
    async def main():
        changes = await StockLivePriceChangeAct().arun(["AAPL"])
        print(changes)

        etf_changes = await ETFLivePriceChangeAct().arun()
        print(etf_changes)

        most_active_stocks = await MostActiveStockersAct().arun()
        print(most_active_stocks)

    asyncio.run(main())
