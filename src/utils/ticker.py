from services.alpaca.api_latest_bars import get_latest_price_bars
# from services.utils import in_memory_cache

INVALID_TICKERS = set()
VALID_TICKERS = set()


# @in_memory_cache(function_name="filter_valid_tickers", ttl=60 * 60)
async def filter_valid_tickers(tickers: list[str]) -> list[str]:
    all_have_been_valid = []

    for ticker in tickers:
        if ticker in INVALID_TICKERS:
            return [ticker]
        if ticker in VALID_TICKERS:
            all_have_been_valid.append(True)
        else:
            all_have_been_valid.append(False)

    if all(all_have_been_valid):
        return []

    invalid_tickers = []
    latest_bars = await get_latest_price_bars(tickers)
    for ticker in tickers:
        if ticker not in latest_bars:
            invalid_tickers.append(ticker)
            INVALID_TICKERS.add(ticker)
        else:
            VALID_TICKERS.add(ticker)
    return invalid_tickers


__all__ = ["filter_valid_tickers"]
