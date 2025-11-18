import json
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from src.services.sandx_ai.api_client import SandxAPIClient
from src.services.utils import APIError, redis_cache

load_dotenv()


PositionItem = TypedDict("PositionItem", {
    "allocation": float,
    "currentPrice": float,
    "ptcChangeInPrice": Annotated[float, "The percentage change in price relative to the open price"],
    "currentValue": Annotated[float, "The total current value of the position in the portfolio"],
    "id": str,
    "ticker": str,
    "volume": int,
    "cost": float,
})


class Position(TypedDict):
    allocation: Annotated[float,
                          "The percentage allocation of the position in the portfolio"]
    current_price: Annotated[float,
                             "The current price of the stock position per share"]
    ptc_change_in_price: Annotated[float,
                                   "The percentage change in price relative to the open price"]
    current_value: Annotated[float,
                             "The total current value of the position in the portfolio"]
    ticker: Annotated[str, "The stock ticker of the position"]
    volume: Annotated[int, "The total share of the position in the portfolio"]
    cost: Annotated[float, "The average cost of the position in the portfolio"]


api_client = SandxAPIClient[list[PositionItem]](
    "/tools/positions")


@redis_cache(ttl=10, function_name="list_positions")
async def list_positions(bot_id: str) -> Sequence[Position]:

    positions = await api_client.get(params={"botId": bot_id})

    readable_positions: list[Position] = []
    for position in positions:
        _dict: Position = {
            "allocation": position["allocation"],
            "current_price": position["currentPrice"],
            "ptc_change_in_price": position["ptcChangeInPrice"],
            "current_value": position["currentValue"],
            "ticker": position["ticker"],
            "volume": position["volume"],
            "cost": position["cost"],
        }
        readable_positions.append(_dict)
    return readable_positions


__all__ = [
    "list_positions",
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
        data = await list_positions(bot_id="7cf5cfb1-b30d-4d82-9363-af2096f2d926")
        print("Positions response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify API_KEY is correct "
            "and has access to the Sandx AI API."
        )

if __name__ == "__main__":

    # python -m src.services.sandx_ai.api_position
    import asyncio
    asyncio.run(_run())
