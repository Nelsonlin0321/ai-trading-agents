from pydantic import BaseModel, Field
from langchain.tools import tool
from src.tools import actions
from src import utils

eft_live_price_change_act = actions.ETFLivePriceChangeAct()
stock_live_price_change_act = actions.StockLivePriceChangeAct()
most_active_stockers_act = actions.MostActiveStockersAct()


@tool(eft_live_price_change_act.name)
async def get_etf_live_historical_price_change():
    """
    Fetch live and historical percent-change metrics for the most-traded U.S. equity ETFs (SPY, QQQ, IWM, etc.) in the different sectors.

    The returned string is a markdown snippet that contains:
    - A level-2 heading
    - A short note explaining the calculation windows (1-day, 1-week, 1-month, 3-month, 1-year, 3-year etc)
    - A table whose columns are derived from the record fields
      (typically including current intraday % change, 1-day, 1-week, 1-month, 3-month, 1-year, 3-year % changes).

    Use this tool when you need a quick, snapshot of ETF momentum and relative strength

    Possible purposes:
    - Compare sector momentum at a glance
    - Identify ETFs with strongest/weakest 1-week or 1-month trends
    - Spot divergences between intraday and longer-term performance
    - Quickly screen for mean-reversion or breakout candidates among liquid ETFs
    - Build relative-strength rankings for portfolio rotation strategies
    """
    results = await eft_live_price_change_act.arun()
    etc_info_dict_list = list(results.values())
    heading = "## ETF Current and Historical Percent Changes"
    note = """
**Note:**
- The percent-change metrics are based on the current trading day and common historical windows such as 1-week, 1-month, and 1-year.
- The current intraday percent is the percent-change of the ETF's current price relative to the previous close."""
    markdown_table = utils.dicts_to_markdown_table(etc_info_dict_list)  # type: ignore
    return heading + "\n\n" + note + "\n\n" + markdown_table


class TickerInput(BaseModel):
    """Input for querying stock current price and historical price changes"""

    tickers: list[str] = Field(
        description="List of stock tickers, e.g. ['AAPL', 'MSFT', 'GOOGL']"
    )


@tool(stock_live_price_change_act.name, args_schema=TickerInput)
async def get_stock_live_historical_price_change(tickers: list[str]):
    """
    Fetch comprehensive percent-change metrics for the provided stock tickers.

    This function queries live and historical price data for each ticker and
    calculates percentage changes over multiple time windows:
    - Current intraday percent change (vs. previous close)
    - 1-day, 1-week, 1-month, 3-month, and 1-year, 3-year percent changes

    The returned data enables quick assessment of momentum, trend strength,
    and relative performance for portfolio monitoring, screening, or market
    commentary.

    Parameters
    ----------
    tickers : list[str]
        A list of valid stock tickers (e.g., ['AAPL', 'MSFT', 'GOOGL']).

    Returns
    -------
    str
        A markdown snippet that contains:
        - A level-2 heading
        - A short note explaining the calculation windows (1-day, 1-week, 1-month, 3-month, 1-year, 3-year)
        - A table whose columns are derived from the record fields
          (typically including current intraday % change, 1-day, 1-week, 1-month, 3-month, 1-year, 3-year % changes).

    Possible purposes:
    - Compare momentum across holdings or watch-list names
    - Identify stocks with strongest/weakest 1-week or 1-month trends
    - Spot divergences between intraday and longer-term performance
    - Screen for mean-reversion or breakout candidates
    - Build relative-strength rankings for sector rotation or portfolio re-balancing
    - Generate quick market commentary on price action
    - Monitor portfolio positions for risk or opportunity signals
    """
    results = await stock_live_price_change_act.arun(tickers)
    metrics_list = []
    for ticker, metrics in results.items():
        _dict = {"ticker": ticker, **metrics}
        metrics_list.append(_dict)

    markdown_table = utils.dicts_to_markdown_table(metrics_list)
    heading = "## Stock Current and Historical Percent Changes"
    datetime = utils.get_current_timestamp()
    note = f"""
**Note:**
- The data is fetched at {datetime} in New York time.
- The percent-change metrics are based on the current trading day and common historical windows such as 1-day, 1-week, 1-month, 3-month, 1-year, and 3-year.
- The current intraday percent is the percent-change of the stock's current price relative to the previous close."""
    return heading + "\n\n" + note + "\n\n" + markdown_table


@tool(most_active_stockers_act.name)
async def get_most_active_stockers():
    """
    Fetch the most active stockers by trading volume in the U.S. stock market.

    The returned string is a markdown snippet that contains:
    - A level-2 heading
    - A short note explaining the calculation windows (1-day, 1-week, 1-month, 3-month, 1-year, 3-year etc)
    - A table whose columns are derived from the record fields
      (typically including current intraday % change, 1-day, 1-week, 1-month, 3-month, 1-year, 3-year % changes).

    Use this tool when you need a quick snapshot of the most active stocks by trading volume in the U.S. market—ideal for spotting liquidity leaders,
    momentum surges, or unusual activity at a glance.

    Possible purposes:
    - Identify high-volume breakouts or breakdowns in real time
    - Screen for momentum-driven day-trading candidates
    - Spot unusual volume spikes that may signal news or earnings reactions
    - Build a liquidity-ranked watch-list for swing or intraday strategies
    - Compare relative volume intensity across leading names
    - Generate quick market commentary on where the action is today
    - Monitor for potential volatility expansion plays based on volume leadership
    """
    results = await most_active_stockers_act.arun()
    last_updated = results["last_updated"]
    most_active_stockers = results["most_actives"]
    markdown_table = utils.dicts_to_markdown_table(most_active_stockers)  # type: ignore
    heading = "## Most Active Stockers"
    note = f"""
**Note:**
- The data is fetched at {last_updated} in New York time.
- The table shows the most active stockers by trading volume in the U.S. stock market.
- The percent-change metrics are based on the current trading day and common historical windows such as 1-day, 1-week, 1-month, 3-month, 1-year, and 3-year.
- The current intraday percent is the percent-change of the stock's current price relative to the previous close.
"""
    return heading + "\n\n" + note + "\n\n" + markdown_table


__all__ = [
    "get_etf_live_historical_price_change",
    "get_stock_live_historical_price_change",
    "get_most_active_stockers",
]
