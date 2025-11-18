from typing import Literal, TypedDict
from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

ActiveStock = TypedDict(
    "ActiveStock", {"symbol": str, "trade_count": int, "volume": int}
)

MostActiveStocksResponse = TypedDict(
    "MostActiveStocksResponse", {"last_updated": str, "most_actives": list[ActiveStock]}
)


mostActiveStocksAPI: AlpacaAPIClient[MostActiveStocksResponse] = AlpacaAPIClient(
    endpoint="/v1beta1/screener/stocks/most-actives"
)


async def get_most_active_stocks(
    by: Literal["trades", "volume"] = "trades", top: int = 20
):
    most_active_stocks = await mostActiveStocksAPI.get(params={"by": by, "top": top})
    return most_active_stocks


__all__ = [
    "get_most_active_stocks",
]


async def _run() -> None:
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()

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
        data = await get_most_active_stocks(by="trades", top=30)
        print("Most active stocks response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
        print(
            f"Symbols returned: {', '.join([stock['symbol'] for stock in data['most_actives']])}"
        )
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_most_active_stockers
    import asyncio

    asyncio.run(_run())
