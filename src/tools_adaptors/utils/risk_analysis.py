from typing import List, TypedDict, NotRequired

import numpy as np

from src import utils
from src.services.alpaca.typing import PriceBar


class VolatilityRisk(TypedDict, total=False):
    error: NotRequired[str]
    volatility_20d: NotRequired[float]
    volatility_60d: NotRequired[float]
    volatility_252d: NotRequired[float]
    garman_klass_volatility: NotRequired[float]
    parkinson_volatility: NotRequired[float]
    realized_volatility: NotRequired[float]
    volatility_clustering: NotRequired[float]
    max_drawdown: NotRequired[float]
    max_drawdown_duration: NotRequired[int]
    var_95: NotRequired[float]
    var_99: NotRequired[float]
    cvar_95: NotRequired[float]
    large_jumps_count: NotRequired[int]
    jump_intensity: NotRequired[float]


def calculate_volatility_risk(
    price_bars: List[PriceBar], lookback_periods: List[int] = [20, 60, 252]
) -> VolatilityRisk:
    """Volatility risk calculation with multiple timeframes and metrics
    price_bars: List of price bars ascending order. the index 0 is the oldest bar,
    and the index -1 is the latest bar.
    lookback_periods: List of lookback periods in days. Default is [20, 60, 252].
    """

    closes = [bar_["close_price"] for bar_ in price_bars]
    highs = [bar_["high_price"] for bar_ in price_bars]
    lows = [bar_["low_price"] for bar_ in price_bars]

    if len(closes) < 2:
        return {"error": "Insufficient data for volatility calculation"}

    # Calculate returns
    returns = np.array(
        [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    )

    results: VolatilityRisk = {}

    # 1. Multi-timeframe Historical Volatility
    for period in lookback_periods:
        if len(returns) >= period:
            period_returns = returns[-period:]
            hist_vol = np.std(period_returns) * np.sqrt(252)
            results[f"volatility_{period}d"] = hist_vol

    # 2. Garman-Klass Volatility (more efficient)
    if len(closes) > 1:
        log_hl = [np.log(high / low) for high, low in zip(highs[1:], lows[1:])]
        log_co = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        gk_vol = np.sqrt(
            np.mean(
                [
                    0.5 * (hl**2) - (2 * np.log(2) - 1) * (co**2)
                    for hl, co in zip(log_hl, log_co)
                ]
            )
        ) * np.sqrt(252)
        results["garman_klass_volatility"] = gk_vol

    # 3. Parkinson Volatility (uses high-low range)
    parkinson_vol = np.sqrt(
        (1 / (4 * np.log(2)))
        * np.mean([np.log(high / low) ** 2 for high, low in zip(highs, lows)])
    ) * np.sqrt(252)
    results["parkinson_volatility"] = parkinson_vol

    # 4. Realized Volatility (RMS of returns)
    realized_vol = np.sqrt(np.mean(returns**2)) * np.sqrt(252)
    results["realized_volatility"] = realized_vol

    # 5. Volatility Clustering (GARCH-like measure)
    squared_returns = returns**2
    volatility_clustering = (
        np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        if len(squared_returns) > 2
        else 0
    )
    results["volatility_clustering"] = volatility_clustering

    # 6. Maximum Drawdown with duration
    peak = closes[0]
    max_drawdown = 0
    # drawdown_start = 0
    max_drawdown_duration = 0
    current_drawdown_duration = 0

    for i, price in enumerate(closes):
        if price > peak:
            peak = price
            current_drawdown_duration = 0
        else:
            drawdown = (peak - price) / peak
            current_drawdown_duration += 1
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_duration = current_drawdown_duration

    results["max_drawdown"] = max_drawdown
    results["max_drawdown_duration"] = max_drawdown_duration

    # 7. Value at Risk (VaR) calculations
    var_95 = np.percentile(returns, 5).item()
    var_99 = np.percentile(returns, 1).item()
    results["var_95"] = var_95
    results["var_99"] = var_99

    # 8. Conditional VaR (Expected Shortfall)
    cvar_95 = (
        np.mean(returns[returns <= var_95]).item()
        if len(returns[returns <= var_95]) > 0
        else var_95
    )
    results["cvar_95"] = cvar_95

    # 10. Jump Detection
    returns_std = np.std(returns)
    large_jumps = len(returns[np.abs(returns) > returns_std * 2])
    results["large_jumps_count"] = large_jumps
    results["jump_intensity"] = large_jumps / len(returns) if len(returns) > 0 else 0

    return results


def calculate_price_risk(
    price_bars: List[PriceBar], lookback_periods: List[int] = [5, 10, 20, 50]
) -> dict:
    """Enhanced price action risk metrics with multiple lookbacks, momentum, and breakout signals
    price_bars: List of price bars ascending order. the index 0 is the oldest bar,
    and the index -1 is the latest bar.
    lookback_periods: List of lookback periods in days. Default is [5, 10, 20, 50].
    """
    if not price_bars:
        return {"error": "No price bars provided"}

    horizons = lookback_periods

    highs = [bar_["high_price"] for bar_ in price_bars]
    lows = [bar_["low_price"] for bar_ in price_bars]
    closes = [bar_["close_price"] for bar_ in price_bars]
    current_price = closes[-1]

    def support_resistance(lookback: int) -> tuple[float, float]:
        """Local support/resistance over lookback"""
        if len(highs) < lookback:
            return min(lows), max(highs)
        return min(lows[-lookback:]), max(highs[-lookback:])

    # Multi-horizon support/resistance
    sr_levels = {h: support_resistance(h) for h in horizons}

    # Distances to nearest levels
    distances = {}
    for h, (sup, res) in sr_levels.items():
        distances[f"support_{h}d"] = (current_price - sup) / current_price
        distances[f"resistance_{h}d"] = (res - current_price) / current_price

    # Momentum and trend strength (dynamically for each horizon)
    momentums = {}
    for h in horizons:
        if len(closes) >= h:
            momentums[f"momentum_{h}d"] = (current_price - closes[-h]) / closes[-h]
        else:
            momentums[f"momentum_{h}d"] = 0

    # Average True Range (proxy for intraday volatility)
    tr = [
        max(high, c_prev) - min(low, c_prev)
        for high, low, c_prev in zip(highs[1:], lows[1:], closes[:-1])
    ]

    atrs = {}
    for h in horizons:
        atrs[f"average_true_range_{h}d"] = (
            np.mean(tr[-h:]) if len(tr) >= h else np.mean(tr) if tr else 0
        )
        atrs[f"average_true_range_{h}d_percent"] = (
            atrs[f"average_true_range_{h}d"] / current_price
        )

    breakouts = {}
    breakdowns = {}
    for h in horizons:
        if h in sr_levels:
            breakouts[f"breakout_{h}d"] = 1 if current_price > sr_levels[h][1] else 0
            breakdowns[f"breakdown_{h}d"] = 1 if current_price < sr_levels[h][0] else 0
        else:
            breakouts[f"breakout_{h}d"] = 0
            breakdowns[f"breakdown_{h}d"] = 0

    # Return enriched dictionary
    return {
        "current_price": current_price,
        **{f"support_{h}d": sr[0] for h, sr in sr_levels.items()},
        **{f"resistance_{h}d": sr[1] for h, sr in sr_levels.items()},
        **distances,
        **momentums,
        **atrs,
        **breakouts,
        **breakdowns,
    }


def format_volatility_risk_markdown(risk, ticker_symbol):
    """
    Simple markdown format with tables for each category
    """
    md = [f"# {ticker_symbol} Volatility Risk Indicators", ""]

    def describe_metric(key: str) -> str:
        if key.startswith("volatility_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"Historical volatility over {horizon} days (annualized)"
        if key == "garman_klass_volatility":
            return "Range-based volatility using open/close and high/low (annualized)"
        if key == "parkinson_volatility":
            return "High–low range volatility estimator (annualized)"
        if key == "realized_volatility":
            return "RMS of daily returns (annualized)"
        if key == "volatility_clustering":
            return "Lag-1 correlation of squared returns (persistence)"
        if key == "max_drawdown":
            return "Largest peak-to-trough decline (fraction of peak)"
        if key == "max_drawdown_duration":
            return "Longest consecutive days spent in drawdown"
        if key == "var_95":
            return "5th percentile daily return (VaR 95%)"
        if key == "var_99":
            return "1st percentile daily return (VaR 99%)"
        if key == "cvar_95":
            return "Average return in worst 5% tail (CVaR 95%)"
        if key == "large_jumps_count":
            return "Days with |return| > 2× std"
        if key == "jump_intensity":
            return "Fraction of days with large jumps"
        return "Indicator"

    md.append("| Metric | Value | Definition |")
    md.append("|--------|-------|------------|")
    for key, value in risk.items():
        md.append(
            f"| {key} | {utils.format_float(value, 3)} | {describe_metric(key)} |"
        )
    md.append("")

    return "\n".join(md)


def format_price_risk_markdown(risk, ticker_symbol):
    """
    Simple markdown format with tables for each category
    """
    md = [f"# {ticker_symbol} Price Risk Indicators", ""]

    def describe_metric(key: str) -> str:
        if key == "current_price":
            return "Latest closing price"
        if key.startswith("momentum_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"Return over {horizon} days relative to prior close"
        if (
            key.startswith("average_true_range_")
            and key.endswith("d")
            and not key.endswith("d_percent")
        ):
            horizon = key.split("_")[3].removesuffix("d")
            return f"Average True Range over {horizon} days (absolute)"
        if key.startswith("average_true_range_") and key.endswith("d_percent"):
            horizon = key.split("_")[3].removesuffix("d")
            return f"ATR over {horizon} days as a fraction of price"
        if key.startswith("support_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"Distance to {horizon}d support (lowest low) as fraction of price"
        if key.startswith("resistance_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return (
                f"Distance to {horizon}d resistance (highest high) as fraction of price"
            )
        if key.startswith("breakout_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"1 if price above {horizon}d resistance; else 0"
        if key.startswith("breakdown_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"1 if price below {horizon}d support; else 0"
        return "Indicator"

    md.append("| Metric | Value | Definition |")
    md.append("|--------|-------|------------|")
    for key, value in risk.items():
        md.append(
            f"| {key} | {utils.format_float(value, 3)} | {describe_metric(key)} |"
        )
    md.append("")

    return "\n".join(md)


__all__ = [
    "calculate_volatility_risk",
    "format_volatility_risk_markdown",
    "calculate_price_risk",
    "format_price_risk_markdown",
]
