from langchain.tools import tool, ToolRuntime
from src.context import Context
from src.tools.actions.portfolio import ListPositionsAct

list_positions_act = ListPositionsAct()


@tool(list_positions_act.name)
async def list_current_positions(runtime: ToolRuntime[Context]):
    """
    Retrieve the current open positions for the trading portfolio, enriched with live market data.

    This tool fetches every active stock position held by the trading portfolio,
    augmenting each record with the latest market price, percentage change since open,
    and the total current market value.  The returned data is ordered by allocation
    (largest first) and can be used to monitor P&L, rebalance allocations, or feed
    downstream analytics, etc.

    Possible tool purposes:
    - Monitor real-time portfolio exposure and sector weightings.
    - Generate daily P&L snapshots for compliance or client reporting.
    - Feed position-level data into risk models (VaR, beta, concentration limits).
    - Detect drift from target allocations and trigger rebalancing alerts.
    - Provide input for tax-loss harvesting by flagging underwater positions.
    - Supply holdings to automated strategies that scale in/out based on allocation caps.
    - Quickly identify top gainers/losers by PnL or PnL percent for intraday reviews.
    - Screen for positions exceeding a PnL percent threshold to automate profit-taking or stop-loss.
    - Aggregate PnL across positions to compute total portfolio return on capital at risk.
    - Export position-level PnL data to Excel for deeper attribution analysis (sector, factor, etc.).
    - Flag large PnL swings outside expected volatility bands for risk-manager alerts.
    - Feed PnL percent into rebalancing algorithms that trim winners and add to laggards.
    - Generate client statements showing dollar and percent gain/loss per holding.
    - Compare realized vs. unrealized PnL to estimate upcoming tax impacts.
    - Provide chatbot answers like “Which positions are up more than 5 % today?” or
      “What is my total unrealized PnL?”

    Notes
    -----
    - Prices reflect the consolidated feed from the exchange with which the
      broker is connected; delays are typically < 500 ms during market hours.
    - CASH is a special position that represents the cash balance in the account.
    - Allocation percentages are computed against the sum of currentValue across
      **all** positions plus any cash held in the same account.
    - Each position now includes `pnl` (dollar profit/loss) and `pnl_percent` (return on cost basis).
    """

    bot_id = runtime.context.bot.id
    position_markdown = await list_positions_act.arun(bot_id=bot_id)
    return position_markdown


__all__ = ["list_current_positions"]
