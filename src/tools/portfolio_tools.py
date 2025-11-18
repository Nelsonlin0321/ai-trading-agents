from typing import Sequence, TypedDict, Annotated
from langchain.tools import tool, ToolRuntime
from src import utils
from src.services.sandx_ai.api_position import list_positions
from src.context import Context
from src.services.sandx_ai.typing import Position


class FormattedPosition(TypedDict):
    allocation: Annotated[str,
                          "The percentage allocation of the position in the portfolio"]
    current_price: Annotated[str,
                             "The current price of the stock position per share"]
    ptc_change_in_price: Annotated[str,
                                   "The percentage change in price relative to the open price"]
    current_value: Annotated[str,
                             "The total current value of the position in the portfolio"]
    ticker: Annotated[str, "The stock ticker of the position"]
    volume: Annotated[str, "The total share of the position in the portfolio"]
    cost: Annotated[str, "The average cost of the position in the portfolio"]


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
            volume=utils.format_float(
                position["volume"], 2),
            cost=utils.format_float(
                position['cost'], 2),
            current_price=utils.format_float(
                position["current_price"], 2),
            ptc_change_in_price=utils.format_percent_change(
                position["ptc_change_in_price"]),
            current_value=utils.format_currency(
                position["current_value"], 2),
            ticker=position["ticker"],
            allocation=utils.format_percent(
                position["allocation"], 3)
        )
        formatted_positions.append(formatted_position)
    position_markdown = utils.dicts_to_markdown_table(formatted_positions)
    heading = "## Current Open Positions"

    datetime = utils.get_current_timestamp()

    note = f"""
    - Datetime New York Time: {datetime}
    - ptcChangeInPrice is the percentage change from the position’s open price to the current price.
    - Allocation percentages are computed against the sum of currentValue across
      **all** positions plus any cash held in the same account.
    """

    return heading + "\n\n" + note + "\n\n" + position_markdown


@tool("Retrieve Current Open Positions")
async def list_positions_tool(runtime: ToolRuntime[Context]):
    """
    Retrieve the current open positions for the trading portfolio, enriched with live market data.

    This tool fetches every active stock position held by the trading portfolio,
    augmenting each record with the latest market price, percentage change since open,
    and the total current market value.  The returned data is ordered by allocation
    (largest first) and can be used to monitor P&L, rebalance allocations, or feed
    downstream analytics, etc.

    Returns
    -------
    Positions Markdown Table
        A markdown table with the following columns:
        allocation : float
            Fraction of the total portfolio market value represented by this position
            (e.g., 0.25 means 25 %).
        currentPrice : float
            Last traded price per share in the account’s currency.
        ptcChangeInPrice : str
            Percentage change from the position’s average open price to the current price,
            formatted with two decimals and a leading “+” or “-” (e.g., “+3.45%”).
        currentValue : float
            Market value of the entire position (currentPrice × volume).
        ticker : str
            Upper-case stock symbol (e.g., “AAPL”).
        volume : int
            Number of shares held (long is positive, short is negative).
        cost : float
            Volume-weighted average cost per share at the time the position was built.
        datetime : str
            ISO-8601 timestamp (YYYY-MM-DDTHH:MM:SS-05:00) indicating when the price
            was last updated in America/New_York time.

    Notes
    -----
    - Prices reflect the consolidated feed from the exchange with which the 
      broker is connected; delays are typically < 500 ms during market hours.
    - Allocation percentages are computed against the sum of currentValue across
      **all** positions plus any cash held in the same account.
    - Short positions are represented with negative volume and may carry a
      negative allocation if the account is net-short.
    """

    bot_id = runtime.context.bot.id
    positions = await list_positions(bot_id)
    position_markdown = convert_positions_to_markdown_table(positions)
    return position_markdown
