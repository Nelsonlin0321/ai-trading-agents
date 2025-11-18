from pydantic import BaseModel, Field
from langchain.tools import tool
from src.tools import actions
from src import utils

eft_full_price_metrics_act = actions.ETFFullPriceMetricsAct()
stock_live_price_change_act = actions.StockLivePriceChangeAct()


@tool(eft_full_price_metrics_act.name)
async def etf_percent_change_summary():
    """
    Fetch live and historical percent-change metrics for the most-traded U.S. equity ETFs (SPY, QQQ, IWM, etc.) in the different sectors.

    The returned string is a markdown snippet that contains:
    - A level-2 heading
    - A short note explaining the calculation windows (1-day, 1-week, 1-month, 1-year)
    - A table whose columns are derived from the record fields
      (typically including current intraday % change, 1-day, 1-week, 1-month, 1-year % changes).

    Use this tool when you need a quick, snapshot of ETF momentum and relative strength
    """
    results = await eft_full_price_metrics_act.arun()
    etc_info_dict_list = list(results.values())
    heading = "## ETF Current and Historical Percent Changes"
    note = """
**Note:**  
- The percent-change metrics are based on the current trading day and common historical windows such as 1-week, 1-month, and 1-year.
- The current intraday percent is the percent-change of the ETF's current price relative to the previous close."""
    markdown_table = utils.dicts_to_markdown_table(
        etc_info_dict_list)  # type: ignore
    return heading + "\n\n" + note + "\n\n" + markdown_table


class TickerInput(BaseModel):
    """Input for querying stock current price and historical price changes  """
    tickers: list[str] = Field(
        description="List of stock tickers, e.g. ['AAPL', 'MSFT', 'GOOGL']")


@tool(stock_live_price_change_act.name, args_schema=TickerInput)
async def stock_live_historical_price_change(tickers: list[str]):
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
