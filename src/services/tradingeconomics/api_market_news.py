import json
from datetime import datetime, timezone
from typing import Sequence, TypedDict

from src.services.tradingeconomics.api_client import TradingEconomicsAPIClient
from src.services.utils import APIError, redis_cache

# https://tradingeconomics.com/ws/stream.ashx?start=15&size=20&c=united states

RawNews = TypedDict(
    "RawNews",
    {
        "ID": int,
        "title": str,
        "description": str,
        "url": str,
        "author": str,
        "country": str,
        "category": str,
        "image": str | None,
        "importance": int,
        "date": str,
        "expiration": str,
        "html": str | None,
        "type": str | None,
    },
)


class News(TypedDict):
    ID: int  # pylint: disable=invalid-name
    title: str
    description: str
    country: str
    category: str
    importance: int
    date: str
    expiration: str
    time_ago: str


api_client: TradingEconomicsAPIClient[Sequence[RawNews]] = TradingEconomicsAPIClient(
    endpoint="/ws/stream.ashx"
)


def _format_relative_time(date_str: str, *, now: datetime | None = None) -> str:  # pylint: disable=too-many-return-statements
    """Return human-friendly relative time (e.g., "4 hours ago").

    Expects ISO-8601 datetime string. If timezone is missing, assumes UTC.
    Handles 'Z' suffix by normalizing to '+00:00'. Falls back gracefully
    to a readable string if parsing fails.
    """
    s = date_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Fallback to a readable format without relative calculation
        return date_str.replace("T", " ")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    ref = now if now is not None else datetime.now(timezone.utc)
    diff_seconds = int((ref - dt).total_seconds())
    future = diff_seconds < 0
    seconds = abs(diff_seconds)

    if seconds < 60:
        return "just now" if not future else "in less than a minute"

    minutes = seconds // 60
    if minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit} ago" if not future else f"in {minutes} {unit}"

    hours = minutes // 60
    if hours < 24:
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit} ago" if not future else f"in {hours} {unit}"

    days = hours // 24
    if days < 7:
        unit = "day" if days == 1 else "days"
        return f"{days} {unit} ago" if not future else f"in {days} {unit}"

    weeks = days // 7
    if weeks < 5:
        unit = "week" if weeks == 1 else "weeks"
        return f"{weeks} {unit} ago" if not future else f"in {weeks} {unit}"

    months = days // 30
    if months < 12:
        unit = "month" if months == 1 else "months"
        return f"{months} {unit} ago" if not future else f"in {months} {unit}"

    years = days // 365
    unit = "year" if years == 1 else "years"
    return f"{years} {unit} ago" if not future else f"in {years} {unit}"


async def get_news() -> Sequence[News]:
    @redis_cache(function_name="tradingeconomics_news", ttl=60 * 5)
    async def _get():
        raw_news_list = await api_client.get(
            params={"c": "united states", "start": 0, "size": 30}, timeout=15.0
        )
        return raw_news_list

    raw_news_list = await _get()
    news_list: Sequence[News] = []
    for new in raw_news_list:
        if new["importance"] > 0:
            news_list.append(
                News(
                    ID=new["ID"],
                    title=new["title"],
                    description=new["description"],
                    country=new["country"],
                    category=new["category"],
                    importance=new["importance"],
                    date=new["date"],
                    expiration=new["expiration"],
                    time_ago=_format_relative_time(new["date"]),
                )
            )
    return news_list


__all__ = [
    "get_news",
]


async def _run() -> None:
    try:
        data = await get_news()
        print("News response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    # python -m src.services.tradingeconomics.api_market_news
    import asyncio

    asyncio.run(_run())
