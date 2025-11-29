from langchain.tools import tool
from src.tools.actions.risk import (
    FundamentalRiskDataAct,
    VolatilityRiskAct,
)


fundamental_risk_act = FundamentalRiskDataAct()
volatility_risk_act = VolatilityRiskAct()


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


@tool(volatility_risk_act.name)
async def get_volatility_risk_indicators(ticker: str):
    """Get volatility and tail-risk indicators for a ticker.

    Fetches ~1 year of daily historical price and compute multi-metric risks

    Includes:
    - Historical volatility (20d, 60d, 252d; annualized)
    - Garman–Klass and Parkinson range-based volatility (annualized)
    - Realized volatility and volatility clustering
    - Maximum drawdown and drawdown duration
    - VaR (95%, 99%) and CVaR (95%)
    - Jump detection (large jumps count, jump intensity)

    Returns a markdown summary of the computed indicators.

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    return await volatility_risk_act.arun(ticker)
