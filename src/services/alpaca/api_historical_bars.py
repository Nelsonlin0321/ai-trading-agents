import json
import os
from typing import Dict, TypedDict, Union

from dotenv import load_dotenv

from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError, in_memory_cache

load_dotenv()

Number = float | int

Bar = TypedDict(
    "Bar",
    {
        "c": Number,
        "h": Number,
        "l": Number,
        "n": int,
        "o": Number,
        "t": str,
        "v": int,
        "vw": Number,
    },
)


class HistoricalBars(TypedDict):
    bars: Dict[str, list[Bar]]
    next_page_token: str | None


historicalBarsAPI: AlpacaAPIClient[HistoricalBars] = AlpacaAPIClient(
    endpoint="/v2/stocks/bars"
)


class PriceBar(TypedDict):
    close_price: Union[int, float]
    high_price: Union[int, float]
    low_price: Union[int, float]
    trade_count: int
    open_price: Union[int, float]
    timestamp: str
    volume: int
    volume_weighted_average_price: Union[int, float]


example = {
    "bars": {
        "AAPL": [
            {
                "c": 272.185,
                "h": 275.24,
                "l": 272.185,
                "n": 66941,
                "o": 275,
                "t": "2025-11-12T05:00:00Z",
                "v": 8039893,
                "vw": 273.512065,
            }
        ],
        "TSLA": [
            {
                "c": 438.255,
                "h": 442.329,
                "l": 436.92,
                "n": 130571,
                "o": 442.15,
                "t": "2025-11-12T05:00:00Z",
                "v": 5538695,
                "vw": 438.975835,
            }
        ],
    },
    "next_page_token": None,
}


def _rename_keys(bars: Dict[str, list[Bar]]) -> Dict[str, list[PriceBar]]:
    new_bars = {}
    for key, bar_list in bars.items():
        new_bars[key] = []
        for _bar in bar_list:
            new_bar = PriceBar(
                close_price=_bar["c"],
                high_price=_bar["h"],
                low_price=_bar["l"],
                trade_count=_bar["n"],
                open_price=_bar["o"],
                timestamp=_bar["t"],
                volume=_bar["v"],
                volume_weighted_average_price=_bar["vw"],
            )
            new_bars[key].append(new_bar)
    return new_bars


@in_memory_cache(function_name="get_historical_price_bars", ttl=60 * 60)
async def _get_price_bar(
    *,
    symbols: list[str],
    timeframe: str,
    start: str,
    end: str,
    page_token=None,
    limit=200,
    sort: str = "asc",
):
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "sort": sort,
        "limit": limit,
    }
    if page_token:
        params["page_token"] = page_token

    api_response = await historicalBarsAPI.get(
        symbols=symbols,
        params=params,
    )
    return api_response


# @redis_cache(function_name="get_historical_price_bars", ttl=3600)
# @in_db_cache(function_name="get_historical_price_bars", ttl=3600)
async def get_historical_price_bars(
    *, symbols: list[str], timeframe: str, start: str, end: str, sort: str = "asc"
) -> Dict[str, list[PriceBar]]:
    """Get historical bars for a list of symbols.

    Args:
        symbols (list[str]): A list of symbols to get historical bars for.
        timeframe (str): The timeframe to get historical bars for.
        start (str): The start time to get historical bars for.
        end (str): The end time to get historical bars for.
        sort (str, optional): The sort order of the historical bars. Defaults to "asc".
        next_page_token (str | None, optional): The next page token to get historical bars for. Defaults to None.

    Returns:
        PriceHistoricalBars: A dictionary of historical bars for each symbol.
    """

    api_response = await _get_price_bar(
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        sort=sort,
    )

    next_page_token = api_response["next_page_token"]
    bars = api_response["bars"]

    while next_page_token is not None:
        next_api_response = await _get_price_bar(
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            sort=sort,
            page_token=next_page_token,
        )

        if next_api_response["bars"]:
            for key, bar_list in next_api_response["bars"].items():
                if key not in bars:
                    bars[key] = []
                bars[key].extend(bar_list)
        next_page_token = next_api_response["next_page_token"]

    bars = _rename_keys(bars)
    return bars


__all__ = ["get_historical_price_bars", "PriceBar"]


async def _run() -> None:
    missing = [
        name
        for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET")
        if not os.environ.get(name)
    ]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this snapshots test."
        )
        return

    try:
        data = await get_historical_price_bars(
            symbols=["AAPL", "MSFT"],
            timeframe="1Day",
            start="2024-11-12T05:00:00Z",
            end="2025-11-12T05:00:00Z",
            sort="desc",
        )
        print("Historical bars response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_historical_bars
    import asyncio

    asyncio.run(_run())
