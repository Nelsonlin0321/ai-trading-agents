from langchain.tools import tool, ToolRuntime
from src.context import Context
from src import db
from src.services.sandx_ai import send_summary_email


@tool("get_user_investment_strategy")
async def get_user_investment_strategy(runtime: ToolRuntime[Context]):
    """
    Retrieve the current investment strategy for the trading portfolio.

    This tool fetches the current investment strategy for the trading portfolio.

    Possible tool purposes:
    - Allow an AI agent to decide which assets or sectors to trade based on the user’s stated risk tolerance or philosophy.
    - Enable dynamic re-allocation logic that switches between conservative, balanced, or aggressive portfolios.
    - Provide context to downstream tools (e.g., stock screeners, rebalancers) so they filter or rank opportunities in line with the user’s mandate.
    - Surface the strategy to a dashboard or chat interface so the user can confirm or update it before orders are placed.
    - Act as a guard-rail that prevents trades violating the strategy (e.g., no crypto for a “dividend-income” strategy).

    Returns
    -------
    Investment Strategy
        A string representing the current investment strategy for the trading portfolio.
    """
    try:
        await db.connect()
        bot_id = runtime.context.bot.id
        bot = await db.prisma.bot.find_unique(where={"id": bot_id})
    except Exception as e:
        raise e
    finally:
        await db.disconnect()

    if not bot:
        raise ValueError(f"Bot with ID {bot_id} not found.")

    return bot.strategy


@tool("send_investment_report_email")
async def send_summary_email_tool(
    investment_summary: str, runtime: ToolRuntime[Context]
):
    """
    Persist and email an HTML-formatted investment summary for the current trading run.

    This tool stores the provided HTML summary in the database record for the active run
    and then triggers an email containing that summary to the user.

    The investment summary must consolidate insights from every analyst involved in the run,
    presenting each analyst’s investment recommendation together with the detailed rationale
    behind it. It must conclude with the final trading decision (buy, sell, or hold) for
    the tickers, again including the rationale that led to that decision.

    Parameters
    ----------
    investment_summary : str
        A professionally formatted, self-contained, and stylish HTML string representing the consolidated investment report.
        It must include fully-styled inline CSS (e.g., font-family, color, padding, borders) to ensure
        consistent rendering across all major email clients. Plain text or Markdown are not acceptable.
        Ensure the content contains:
        - Aggregated insights from all analysts to comprehensively evaluate the tickers.
        - Each analyst’s recommendation and its rationale.
        - Final trading actions (buy/sell/hold) with supporting rationale for each ticker.
        - Line charts and bar charts rendered purely with HTML and CSS (no external images or scripts).
    """
    try:
        await db.connect()
        run_id = runtime.context.run.id
        await db.prisma.run.update(
            where={"id": run_id}, data={"summary": investment_summary}
        )
        response = await send_summary_email(run_id)
        return response
    except Exception as e:
        raise e
    finally:
        await db.disconnect()
