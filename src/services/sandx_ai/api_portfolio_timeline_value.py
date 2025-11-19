import json
import os
from datetime import datetime, timedelta
from typing import Sequence, TypedDict
from dotenv import load_dotenv
from src.services.sandx_ai.api_client import SandxAPIClient
from src.services.utils import APIError, redis_cache

load_dotenv()


TimelineValue = TypedDict(
    "TimelineValue",
    {
        "date": str,
        "value": float,
    },
)


api_client = SandxAPIClient[list[TimelineValue]]("/tools/portfolio/timeline-value")


@redis_cache(ttl=10, function_name="get_timeline_values")
async def get_timeline_values(bot_id: str) -> Sequence[TimelineValue]:
    from_date = datetime.now() - timedelta(days=365) - timedelta(days=7)
    from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
    timeline_values = await api_client.get(
        params={"botId": bot_id, "from": from_date_str}
    )
    return timeline_values


__all__ = [
    "get_timeline_values",
]


async def _run() -> None:
    missing = [name for name in ["API_KEY"] if not os.environ.get(name)]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this positions test."
        )
        return

    try:
        data = await get_timeline_values(bot_id="7cf5cfb1-b30d-4d82-9363-af2096f2d926")
        print("Timeline values response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify API_KEY is correct "
            "and has access to the Sandx AI API."
        )


if __name__ == "__main__":
    # python -m src.services.sandx_ai.api_portfolio_timeline_value
    import asyncio

    asyncio.run(_run())
