from langchain.tools import tool
from src.tools.actions import GoogleMarketResearchAct

google_market_research_action = GoogleMarketResearchAct()


@tool(google_market_research_action.name)
async def google_market_research():
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
    market_research = await google_market_research_action.arun()
    return market_research


__all__ = ["google_market_research"]
