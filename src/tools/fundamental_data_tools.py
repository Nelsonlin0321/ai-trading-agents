from langchain.tools import tool
from src.tools.actions.fundamental_data import (
    FundamentalDataAct,
    FundamentalRiskDataAct,
)

fundamental_act = FundamentalDataAct()
fundamental_risk_act = FundamentalRiskDataAct()


@tool(fundamental_act.name)
async def get_fundamental_data(ticker: str):
    """Get fundamental data for a given ticker.

    Returns a markdown report structured into decision-ready sections:
    - Valuation Metrics
    - Profitability & Margins
    - Financial Health & Liquidity
    - Growth Metrics
    - Dividend & Payout
    - Market & Trading Data
    - Analyst Estimates & Ratings
    - Company Information
    - Ownership & Shares
    - Risk & Volatility
    - Technical Indicators

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    return await fundamental_act.arun(ticker)


@tool(fundamental_risk_act.name)
async def get_fundamental_risk_data(ticker: str):
    """Get fundamental risk data for a given ticker.

    Returns a markdown report structured into decision-ready sections:
    - Valuation Metrics
    - Profitability & Margins
    - Financial Health & Liquidity
    - Growth Metrics
    - Dividend & Payout
    - Market & Trading Data
    - Analyst Estimates & Ratings
    - Company Information
    - Ownership & Shares
    - Risk & Volatility
    - Technical Indicators

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    return await fundamental_risk_act.arun(ticker)
