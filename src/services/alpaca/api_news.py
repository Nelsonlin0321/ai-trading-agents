import json
import os
from typing import Dict, TypedDict

from dotenv import load_dotenv

from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

load_dotenv()

example = {
    "author": "Anusuya Lahiri",
    "content": "...",
    "created_at": "2025-11-28T18:25:15Z",
    "headline": "Intel Stock Soars 7% On Report It Could Soon Build Chips For Apple's Macs",
    "id": 49121202,
    "images": [
        {
            "size": "large",
            "url": "https://cdn.benzinga.com/files/imagecache/2048x1536xUP/images/story/2025/11/28/Intel-Corp.jpeg",
        },
        {
            "size": "small",
            "url": "https://cdn.benzinga.com/files/imagecache/1024x768xUP/images/story/2025/11/28/Intel-Corp.jpeg",
        },
        {
            "size": "thumb",
            "url": "https://cdn.benzinga.com/files/imagecache/250x187xUP/images/story/2025/11/28/Intel-Corp.jpeg",
        },
    ],
    "source": "benzinga",
    "summary": "Intel&#39;s stock gains momentum as Apple considers it as a supplier for their next-gen M chips, signaling a potential partnership.",
    "symbols": ["AAPL", "INTC", "TSM"],
    "updated_at": "2025-11-28T18:25:16Z",
    "url": "https://www.benzinga.com/analyst-stock-ratings/analyst-color/25/11/49121202/intel-stock-soars-7-on-report-it-could-soon-build-chips-for-apples-macs",
}

Image = TypedDict(
    "Image",
    {
        "size": str,
        "url": str,
    },
)

News = TypedDict(
    "News",
    {
        "id": int,
        "author": str,
        "content": str,
        "created_at": str,
        "headline": str,
        "images": list[Image],
        "source": str,
        "summary": str,
        "symbols": list[str],
        "updated_at": str,
        "url": str,
    },
)


class NewsResponse(TypedDict):
    news: Dict[str, list[News]]


NewsAPI: AlpacaAPIClient[NewsResponse] = AlpacaAPIClient(endpoint="/v1beta1/news")

# @redis_cache(function_name="get_historical_price_bars", ttl=3600)
# @in_db_cache(function_name="get_historical_price_bars", ttl=3600)


async def get_news(
    symbols: list[str],
    start: str,
    end: str,
    sort: str = "desc",
    limit: int = 12,
) -> Dict[str, list[News]]:
    response = await NewsAPI.get(
        params={
            "symbols": ",".join(symbols),
            "start": start,
            "end": end,
            "sort": sort,
            "limit": limit,
            "include_content": True,
        }
    )
    return response["news"]


__all__ = ["get_news", "News"]


async def _run() -> None:
    from datetime import date, timedelta

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
        start = (date.today() - timedelta(days=5)).isoformat()
        end = date.today().isoformat()
        data = await get_news(
            symbols=["AAPL", "MSFT"],
            start=start,
            end=end,
            sort="desc",
        )
        print("News response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_news
    import asyncio

    asyncio.run(_run())
