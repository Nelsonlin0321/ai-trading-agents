import yfinance as yf
from src.utils import async_wrap
from src.services.utils import redis_cache


@async_wrap
def get_ticker_info(ticker: str) -> dict:
    yf_ticker = yf.Ticker(ticker=ticker)
    info = yf_ticker.info
    return info


@redis_cache(function_name="get_ticker_info", ttl=60 * 60 * 24)
async def async_get_ticker_info(ticker: str) -> dict:
    info = await get_ticker_info(ticker)  # type: ignore
    return info
