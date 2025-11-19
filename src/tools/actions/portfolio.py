from typing import Sequence, Annotated, TypedDict
from src.tools.actions import Action
from src.services.sandx_ai import list_positions, get_timeline_values
from src.services.sandx_ai.typing import Position
from src.tools.actions import utils as action_utils
from src import utils


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
        return "List Current Open Positions"

    async def arun(self, bot_id: str) -> str:
        positions = await list_positions(bot_id)
        position_markdown = convert_positions_to_markdown_table(positions)
        return position_markdown


class PortfolioPerformanceAnalysisAct(Action):
    @property
    def name(self):
        return "Portfolio Performance Analysis"

    async def arun(self, bot_id: str):
        timeline_values = await get_timeline_values(bot_id)
        analysis = action_utils.analyze_timeline_value(timeline_values)
        if analysis:
            return action_utils.create_performance_narrative(analysis)

        return "Insufficient data for analysis."


__all__ = ["ListPositionsAct", "PortfolioPerformanceAnalysisAct"]
