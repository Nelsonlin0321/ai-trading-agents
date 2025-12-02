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


Quote = TypedDict(
    "Quote",
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

QuotesResponse = TypedDict(
    "QuotesResponse",
    {
        "quotes": Dict[str, Quote],
    },
)

QuoteHuman = TypedDict(
    "QuoteHuman",
    {
        "ask_price": Union[int, float],
        "ask_size": int,
        "ask_exchange": str,
        "bid_price": Union[int, float],
        "bid_size": int,
        "bid_exchange": str,
        "conditions": list[str],
        "timestamp": str,
        "market_center": str,
    },
)

QuotesHumanResponse = TypedDict(
    "QuotesHumanResponse",
    {
        "quotes": Dict[str, QuoteHuman],
    },
)


def _rename_key(quote: Quote):
    quote_human: QuoteHuman = {
        "ask_price": quote["ap"],
        "ask_size": quote["as"],
        "ask_exchange": quote["ax"],
        "bid_price": quote["bp"],
        "bid_size": quote["bs"],
        "bid_exchange": quote["bx"],
        "conditions": quote["c"],
        "timestamp": quote["t"],
        "market_center": quote["z"],
    }
    return quote_human


latestQuotesAPI: AlpacaAPIClient[QuotesResponse] = AlpacaAPIClient(
    endpoint="/v2/stocks/quotes/latest"
)


async def get_latest_quotes(
    symbols: list[str],
):
    api_response = await latestQuotesAPI.get(
        symbols=symbols,
    )
    response: QuotesHumanResponse = {
        "quotes": {
            symbol: _rename_key(quote)
            for symbol, quote in api_response["quotes"].items()
        }
    }

    return response


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
            symbols=["AAPL"],
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
    # python -m src.services.alpaca.api_latest_quotes
    import asyncio

    asyncio.run(_run())
