from langchain.tools import tool
from src.utils.ticker import is_valid_ticker
from src.tools_adaptors.risk import (
    FundamentalRiskDataAct,
    VolatilityRiskAct,
    PriceRiskAct,
)


fundamental_risk_act = FundamentalRiskDataAct()
volatility_risk_act = VolatilityRiskAct()
price_risk_act = PriceRiskAct()


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
    symbol = ticker.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"
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
    symbol = ticker.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"
    return await volatility_risk_act.arun(ticker)


@tool(price_risk_act.name)
async def get_price_risk_indicators(ticker: str):
    """Get price action risk indicators for a ticker.

    Computes multi-horizon support/resistance, momentum, ATR, and breakout signals
    from ~1 year of daily bars.

    Includes:
    - Support/Resistance over lookbacks (default: 5, 10, 20, 50 days)
    - Distance to nearest support/resistance normalized by `current_price(T-1)`
    - Momentum per lookback: `(current_price - close[-h]) / close[-h]`
    - Average True Range per lookback and percent of price
    - Breakout/Breakdown flags based on support/resistance breaches
    - `current_price(T-1)`

    Returns a markdown table summarizing the computed indicators.

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    symbol = ticker.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"
    return await price_risk_act.arun(ticker)


__all__ = [
    "get_fundamental_risk_data",
    "get_volatility_risk_indicators",
    "get_price_risk_indicators",
]
