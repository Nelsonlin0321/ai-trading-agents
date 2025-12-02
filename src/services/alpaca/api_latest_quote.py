import json
import os
from typing import Dict, TypedDict, Union

from dotenv import load_dotenv

from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

load_dotenv()

example = {
    "quotes": {
        "AAPL": {
            "ap": 283.18,
            "as": 200,
            "ax": "P",
            "bp": 283.01,
            "bs": 21700,
            "bx": "P",
            "c": ["R"],
            "t": "2025-12-02T00:59:50.364749005Z",
            "z": "C",
        }
    }
}


LatestQuote = TypedDict(
    "LatestQuote",
    {
        "ap": Union[int, float],
        "as": int,
        "ax": str,
        "bp": Union[int, float],
        "bs": int,
        "bx": str,
        "c": list[str],
        "t": str,
        "z": str,
    },
)

LatestQuotes = Dict[str, LatestQuote]


latestQuotesAPI: AlpacaAPIClient[LatestQuotes] = AlpacaAPIClient(
    endpoint="/v2/stocks/quotes/latest"
)


async def get_latest_quotes(
    symbols: list[str],
):
    api_response = await latestQuotesAPI.get(
        symbols=symbols,
    )
    return api_response


__all__ = ["get_latest_quotes"]


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
        data = await get_latest_quotes(
            symbols=["AAPL", "MSFT"],
        )
        print("Latest quotes response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_latest_quote
    import asyncio

    asyncio.run(_run())
