from services.alpaca.api_latest_bars import get_latest_price_bars
from services.utils import in_memory_cache


@in_memory_cache(function_name="filter_valid_tickers", ttl=60 * 60)
async def filter_valid_tickers(tickers: list[str]) -> list[str]:
    invalid_tickers = []
    latest_bars = await get_latest_price_bars(tickers)
    for ticker in tickers:
        if ticker not in latest_bars:
            invalid_tickers.append(ticker)
    return invalid_tickers


__all__ = ["filter_valid_tickers"]
