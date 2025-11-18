from typing import Dict, TypedDict
from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

Number = float | int

DailyBar = TypedDict(
    "DailyBar",
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

MinuteBar = TypedDict(
    "MinuteBar",
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

LatestQuote = TypedDict(
    "LatestQuote",
    {
        "ap": Number,
        "as": int,
        "ax": str,
        "bp": Number,
        "bs": int,
        "bx": str,
        "c": list[str],
        "t": str,
        "z": str,
    },
)

LatestTrade = TypedDict(
    "LatestTrade",
    {
        "c": list[str],
        "i": int,
        "p": Number,
        "s": int,
        "t": str,
        "x": str,
        "z": str,
    },
)

Snapshot = TypedDict(
    "Snapshot",
    {
        "dailyBar": DailyBar,
        "latestQuote": LatestQuote,
        "latestTrade": LatestTrade,
        "minuteBar": MinuteBar,
        "prevDailyBar": DailyBar,
    },
)

SnapshotsResponse = Dict[str, Snapshot]


snapshotsAPI: AlpacaAPIClient[SnapshotsResponse] = AlpacaAPIClient(
    endpoint="/v2/stocks/snapshots"
)


async def get_snapshots(symbols: list[str]) -> SnapshotsResponse:
    return await snapshotsAPI.get(symbols=symbols)

__all__ = [
    "get_snapshots",
]


async def _run() -> None:
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()

    missing = [name for name in (
        "ALPACA_API_KEY", "ALPACA_API_SECRET") if not os.environ.get(name)]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this snapshots test."
        )
        return

    try:
        data = await get_snapshots(symbols=["AAPL", "MSFT"])
        print("Snapshots response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
        print(f"Symbols returned: {', '.join(list(data.keys()))}")
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_snapshots
    import asyncio
    asyncio.run(_run())
