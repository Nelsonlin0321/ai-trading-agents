from typing import Sequence, Annotated, TypedDict
import traceback
from loguru import logger
from prisma.types import PositionCreateInput, PositionUpdateInput
from src.tools.actions import Action
from src.services.sandx_ai import list_positions, get_timeline_values
from src.services.sandx_ai.typing import Position
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client
from src.services.alpaca import get_latest_quotes
from src.tools.actions import utils as action_utils
from src import utils, db


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

    async def arun(self, bot_id: str) -> str:
        positions = await list_positions(bot_id)
        position_markdown = convert_positions_to_markdown_table(positions)
        return position_markdown


class PortfolioPerformanceAnalysisAct(Action):
    @property
    def name(self):
        return "get_portfolio_performance_analysis"

    async def arun(self, bot_id: str):
        timeline_values = await get_timeline_values(bot_id)
        analysis = action_utils.analyze_timeline_value(timeline_values)
        if analysis:
            return action_utils.create_performance_narrative(analysis)

        return "Insufficient data for analysis."


class BuyAct(Action):
    @property
    def name(self):
        return "buy_stock"

    async def arun(self, bot_id: str, ticker: str, volume: float):
        try:
            clock = alpaca_trading_client.get_clock()
            if not clock.is_open:  # type: ignore
                return "Market is closed. Cannot buy stock."
            quotes = await get_latest_quotes([ticker])
            price = quotes["quotes"].get(ticker, {}).get("ask_price")
            if not price:
                return f"Cannot get price for {ticker}"
            price = float(price)
            total_cost = price * volume

            await db.connect()
            async with db.prisma.tx() as transaction:
                portfolio = await transaction.portfolio.find_unique(
                    where={"botId": bot_id}
                )
                if portfolio is None:
                    raise ValueError("Portfolio not found")
                if portfolio.cash < total_cost:
                    return f"Not enough cash to buy {volume} shares of {ticker} at {price} per share."
                portfolio.cash -= total_cost
                await transaction.portfolio.update(
                    where={"botId": bot_id}, data={"cash": portfolio.cash}
                )
                existing = await transaction.position.find_unique(
                    where={
                        "portfolioId_ticker": {
                            "portfolioId": portfolio.id,
                            "ticker": ticker,
                        }
                    }
                )

                if existing is None:
                    await transaction.position.create(
                        data=PositionCreateInput(
                            ticker=ticker,
                            volume=volume,
                            portfolioId=portfolio.id,
                            cost=price,
                        )
                    )
                else:
                    await transaction.position.update(
                        where={
                            "portfolioId_ticker": {
                                "portfolioId": portfolio.id,
                                "ticker": ticker,
                            }
                        },
                        data=PositionUpdateInput(
                            volume=existing.volume + volume,
                            cost=(existing.cost * existing.volume + price * volume)
                            / (existing.volume + volume),
                        ),
                    )

                    return (
                        f"Successfully bought {volume} shares of {ticker} at {price} per share. "
                        f"Current volume is {existing.volume + volume} "
                        f"with average cost {utils.format_float(existing.cost)}"
                    )

        except Exception as e:
            logger.error(f"Error buying stock: {e} Traceback: {traceback.format_exc()}")
            return f"Failed to buy {volume} shares of {ticker}"
        finally:
            await db.disconnect()


__all__ = ["ListPositionsAct", "PortfolioPerformanceAnalysisAct", "BuyAct"]
