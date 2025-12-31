from langchain.tools import tool
from langchain.tools import ToolRuntime
from prisma.types import EquityResearchCreateInput, MarketResearchCreateInput
from src import db
from src.context import Context

# from src.utils.ticker import is_valid_ticker
from src.tools_adaptors.research import GoogleMarketResearchAct, GoogleEquityResearchAct

google_market_research_action = GoogleMarketResearchAct()
google_equity_research_action = GoogleEquityResearchAct()


@tool(google_market_research_action.name)
async def do_google_market_research(runtime: ToolRuntime[Context]):
    """
    Performs market research to synthesize and present the comprehensive current market narrative,
    key drivers, risks, and opportunities based on real-time data and recent news using Google’s grounded LLM.
    returns up-to-date market insights relevant to that strategy.
    Use this tool when you need to:
    - Understand the current market landscape and sentiment
    - Identify key macro and micro drivers moving markets
    - Surface latent risks (geopolitical, regulatory, earnings, etc.)
    - Spot emerging opportunities across sectors or asset classes
    - Obtain a concise, evidence-based narrative for portfolio positioning
    - Make informed investment or allocation decisions grounded in the latest public information
    """
    # user_investment_strategy = runtime.context.bot.strategy

    runId = runtime.context.run.id
    existed = await db.prisma.equityresearch.find_first(
        where={
            "runId": runId,
        }
    )
    if existed:
        return existed.research

    market_research = await google_market_research_action.arun()
    await db.prisma.marketresearch.create(
        data=MarketResearchCreateInput(
            runId=runId,
            research=market_research,
        )
    )
    return market_research


@tool(google_equity_research_action.name)
async def do_google_equity_research(ticker: str, runtime: ToolRuntime[Context]):
    """
    Performs equity research to synthesize and present the comprehensive current market narrative,
    key drivers, risks, and opportunities based on real-time data and recent news using Google’s grounded LLM.
    Returns up-to-date equity insights relevant to that strategy.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA') for which to generate research.
    """
    # symbol = ticker.upper().strip()
    # is_valid = await is_valid_ticker(symbol)
    # if not is_valid:
    #     return f"{symbol} is an invalid ticker symbol"

    runId = runtime.context.run.id
    existed = await db.prisma.equityresearch.find_first(
        where={
            "runId": runId,
            "ticker": ticker,
        }
    )
    if existed:
        return existed.research
    equity_research = await google_equity_research_action.arun(ticker)
    await db.prisma.equityresearch.create(
        data=EquityResearchCreateInput(
            runId=runId,
            ticker=ticker,
            research=equity_research,
        )
    )
    return equity_research


__all__ = ["do_google_market_research", "do_google_equity_research"]
