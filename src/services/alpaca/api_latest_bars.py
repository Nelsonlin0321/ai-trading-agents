from typing import Dict, TypedDict

from dotenv import load_dotenv

from services.alpaca.api_historical_bars import PriceBar
from src.services.alpaca.api_client import AlpacaAPIClient

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


class LatestBars(TypedDict):
    bars: Dict[str, Bar]


latestBarsAPI: AlpacaAPIClient[LatestBars] = AlpacaAPIClient(
    endpoint="/v2/stocks/bars/latest"
)


example = {
    "bars": {
        "AAPL": {
            "c": 272.185,
            "h": 275.24,
            "l": 272.185,
            "n": 66941,
            "o": 275,
            "t": "2025-11-12T05:00:00Z",
            "v": 8039893,
            "vw": 273.512065,
        },
        "TSLA": {
            "c": 438.255,
            "h": 442.329,
            "l": 436.92,
            "n": 130571,
            "o": 442.15,
            "t": "2025-11-12T05:00:00Z",
            "v": 5538695,
            "vw": 438.975835,
        },
    }
}


def _rename_keys(bars: Dict[str, Bar]) -> Dict[str, PriceBar]:
    new_bars = {}
    for key, bar in bars.items():
        new_bar = PriceBar(
            close_price=bar["c"],
            high_price=bar["h"],
            low_price=bar["l"],
            trade_count=bar["n"],
            open_price=bar["o"],
            timestamp=bar["t"],
            volume=bar["v"],
            volume_weighted_average_price=bar["vw"],
        )
        new_bars[key] = new_bar
    return new_bars


async def get_latest_price_bars(symbols: list[str]) -> Dict[str, PriceBar]:
    response = await latestBarsAPI.get(
        symbols=symbols,
    )
    bars = response["bars"]
    latest_bars = _rename_keys(bars=bars)
    return latest_bars
