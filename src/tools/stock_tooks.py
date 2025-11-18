from langchain.tools import tool
from src.tools.actions import ETFFullPriceMetricsAct
from src import utils

eft_full_price_metrics_act = ETFFullPriceMetricsAct()


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
