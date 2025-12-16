from typing import Sequence, Annotated, TypedDict
from datetime import datetime
import pytz
from src.tools_adaptors.base import Action
from src.services.sandx_ai import get_timeline_values
from src.tools_adaptors import utils as action_utils
from src import utils
from src.utils import async_retry
from src.db import prisma
from src.services.alpaca import get_snapshots


class Position(TypedDict):
    ticker: str
    allocation: float
    current_price: float
    ptc_change_in_price: float
    current_value: float
    volume: float
    cost: float
    pnl: float
    pnl_percent: float


async def get_latest_positions(bot_id: str) -> list[Position]:
    portfolio = await prisma.portfolio.find_unique(
        where={"botId": bot_id},
        include={"positions": True},
    )

    if not portfolio:
        return []

    positions = portfolio.positions
    if positions is None:
        positions = []

    cash = portfolio.cash
    tickers = [p.ticker for p in positions]

    position_with_value_ptc: list[Position] = []

    if not tickers:
        position_with_value_ptc.append(
            {
                "allocation": 1.0,
                "current_price": 1.0,
                "ptc_change_in_price": 0.0,
                "current_value": cash,
                "ticker": "CASH",
                "volume": cash,
                "cost": cash,
                "pnl": 0.0,
                "pnl_percent": 0.0,
            }
        )
        return position_with_value_ptc

    tickers_snapshot = await get_snapshots(tickers)

    position_with_value = []

    now = datetime.now()
    ny_time = now.astimezone(pytz.timezone("America/New_York"))
    ny_hour = ny_time.hour
    ny_minute = ny_time.minute

    use_daily_bar = ny_hour < 9 or (ny_hour == 9 and ny_minute < 30)

    for position in positions:
        snapshot = tickers_snapshot.get(position.ticker)
        if not snapshot:
            continue

        latest_quote = snapshot["latestQuote"]
        daily_bar = snapshot["dailyBar"]
        prev_daily_bar = snapshot["prevDailyBar"]

        base_bar = daily_bar if use_daily_bar else prev_daily_bar

        # Fallback if base_bar is missing (e.g. new listing, data issue)
        # Using 0 or some default to avoid division by zero or key errors
        base_close = base_bar["c"] if base_bar else (daily_bar["c"] if daily_bar else 0)

        bid_price = latest_quote["bp"]

        ptc_change_in_price = (
            (bid_price - base_close) / base_close
            if base_close and base_close > 0
            else 0
        )

        current_value = position.volume * bid_price
        pnl = current_value - (position.volume * position.cost)
        pnl_percent = (
            (bid_price - position.cost) / position.cost if position.cost > 0 else 0
        )

        position_with_value.append(
            {
                "ticker": position.ticker,
                "volume": position.volume,
                "cost": position.cost,
                "current_price": bid_price,
                "ptc_change_in_price": ptc_change_in_price,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
            }
        )

    position_total_value = sum(p["current_value"] for p in position_with_value) + cash

    for p in position_with_value:
        position_with_value_ptc.append(
            {
                "ticker": p["ticker"],
                "volume": p["volume"],
                "cost": p["cost"],
                "current_price": p["current_price"],
                "ptc_change_in_price": p["ptc_change_in_price"],
                "current_value": p["current_value"],
                "pnl": p["pnl"],
                "pnl_percent": p["pnl_percent"],
                "allocation": p["current_value"] / position_total_value
                if position_total_value > 0
                else 0,
            }
        )

    position_with_value_ptc.append(
        {
            "allocation": cash / position_total_value
            if position_total_value > 0
            else 0,
            "current_price": 1.0,
            "ptc_change_in_price": 0.0,
            "current_value": cash,
            "ticker": "CASH",
            "volume": cash,
            "cost": cash,
            "pnl": 0.0,
            "pnl_percent": 0.0,
        }
    )

    return position_with_value_ptc


class FormattedPosition(TypedDict):
    ticker: Annotated[str, "The stock ticker of the position"]
    allocation: Annotated[
        str, "The percentage allocation of the position in the portfolio"
    ]
    current_price: Annotated[str, "The current price of the stock position per share"]
    ptc_change_in_price: Annotated[
        str, "The percentage change in price relative to the open price"
    ]
    current_value: Annotated[
        str, "The total current value of the position in the portfolio"
    ]
    volume: Annotated[str, "The total share of the position in the portfolio"]
    cost: Annotated[str, "The average cost of the position in the portfolio"]
    pnl: Annotated[str, "Profit and Loss of the position in the portfolio"]
    pnl_percent: Annotated[
        str, "Profit and Loss percentage of the position in the portfolio"
    ]


def convert_positions_to_markdown_table(positions: Sequence[Position]) -> str:
    """
    Convert a list of Position objects to a markdown table.

    Parameters
    ----------
    positions : list[Position]
        A list of Position objects to be converted into a markdown table.

    Returns
    -------
    str
        A markdown table string representation of the positions.
    """
    positions = sorted(positions, key=lambda x: x["allocation"], reverse=True)
    formatted_positions = []
    for position in positions:
        formatted_position = FormattedPosition(
            ticker=position["ticker"],
            volume=utils.format_float(position["volume"]),
            cost=utils.format_float(position["cost"]),
            current_price=utils.format_float(position["current_price"]),
            ptc_change_in_price=utils.format_percent_change(
                position["ptc_change_in_price"]
            ),
            current_value=utils.format_currency(position["current_value"]),
            allocation=utils.format_percent(position["allocation"]),
            pnl=utils.format_currency(position["pnl"]),
            pnl_percent=utils.format_percent(position["pnl_percent"]),
        )
        formatted_positions.append(formatted_position)
    position_markdown = utils.dicts_to_markdown_table(formatted_positions)
    heading = "## User's Current Open Positions"

    datetime = utils.get_current_timestamp()

    note = f"""
- Datetime New York Time: {datetime}
- CASH is a special position that represents the cash balance in the account.
- ptc_change_in_price is the percentage change from the position’s open price to the current price.
- Allocation percentages are computed against the sum of currentValue across
    **all** positions plus any cash held in the same account.
    """

    return heading + "\n\n" + note + "\n\n" + position_markdown


class ListPositionsAct(Action):
    @property
    def name(self):
        return "list_current_positions"

    @async_retry()
    async def arun(self, bot_id: str) -> str:
        positions = await get_latest_positions(bot_id)
        position_markdown = convert_positions_to_markdown_table(positions)
        return position_markdown


class PortfolioPerformanceAnalysisAct(Action):
    @property
    def name(self):
        return "get_portfolio_performance_analysis"

    @async_retry()
    async def arun(self, bot_id: str):
        timeline_values = await get_timeline_values(bot_id)
        analysis = action_utils.analyze_timeline_value(timeline_values)
        if analysis:
            return action_utils.create_performance_narrative(analysis)

        return "Insufficient data for analysis."


__all__ = ["ListPositionsAct", "PortfolioPerformanceAnalysisAct"]


if __name__ == "__main__":
    #  python -m src.tools_adaptors.portfolio]
    import asyncio

    async def arun(bot_id: str):
        from src import db

        await db.connect()
        positions = await get_latest_positions(bot_id)
        position_markdown = convert_positions_to_markdown_table(positions)
        print(position_markdown)
        await db.disconnect()

    asyncio.run(arun("d4ef2264-31fa-438a-92b0-5c3b00db8325"))
