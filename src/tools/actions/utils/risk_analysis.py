from typing import Dict, List

import numpy as np

from src.services.alpaca.typing import PriceBar


def calculate_enhanced_volatility_risk(
    price_bars: List[PriceBar], lookback_periods: List[int] = [20, 60, 252]
) -> Dict:
    """Enhanced volatility risk calculation with multiple timeframes and metrics"""

    closes = [bar_["close_price"] for bar_ in price_bars]
    highs = [bar_["high_price"] for bar_ in price_bars]
    lows = [bar_["low_price"] for bar_ in price_bars]

    if len(closes) < 2:
        return {"error": "Insufficient data for volatility calculation"}

    # Calculate returns
    returns = np.array(
        [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    )

    results = {}

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
    var_95 = np.percentile(returns, 5)  # 5% worst return
    var_99 = np.percentile(returns, 1)  # 1% worst return
    results["var_95"] = var_95
    results["var_99"] = var_99

    # 8. Conditional VaR (Expected Shortfall)
    cvar_95 = (
        np.mean(returns[returns <= var_95])
        if len(returns[returns <= var_95]) > 0
        else var_95
    )
    results["cvar_95"] = cvar_95

    # 9. Volatility Regime Detection
    short_term_vol = (
        np.std(returns[-min(20, len(returns)) :]) * np.sqrt(252)
        if len(returns) >= 5
        else 0
    )
    long_term_vol = np.std(returns) * np.sqrt(252)

    if short_term_vol > long_term_vol * 1.5:
        results["volatility_regime"] = "High Volatility"
    elif short_term_vol < long_term_vol * 0.7:
        results["volatility_regime"] = "Low Volatility"
    else:
        results["volatility_regime"] = "Normal Volatility"

    # 10. Jump Detection
    returns_std = np.std(returns)
    large_jumps = len(returns[np.abs(returns) > returns_std * 2])
    results["large_jumps_count"] = large_jumps
    results["jump_intensity"] = large_jumps / len(returns) if len(returns) > 0 else 0

    return results
