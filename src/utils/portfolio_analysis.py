from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from prisma.models import DailyPortfolioSnapshot


def create_multi_period_performance_analysis(
    snapshots: List[DailyPortfolioSnapshot],
) -> Dict[str, Any]:
    """
    Create comprehensive performance analysis across multiple time periods
    Skips periods with insufficient data
    """
    if not snapshots:
        return {"error": "No snapshot data available"}

    # Sort by date to ensure chronological order
    sorted_snapshots = sorted(snapshots, key=lambda x: x.date)
    dates = [s.date for s in sorted_snapshots]
    portfolio_values = [s.portfolioValue for s in sorted_snapshots]

    if len(sorted_snapshots) < 2:
        return {"error": "Insufficient data for analysis"}

    current_date = sorted_snapshots[-1].date
    current_value = portfolio_values[-1]

    def calculate_period_metrics(
        start_index: int, end_index: int, period_name: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate metrics for a specific period, return None if insufficient data"""
        if start_index < 0 or start_index >= end_index:
            return None

        # Require at least 2 data points for meaningful analysis
        if end_index - start_index < 1:
            return None

        start_value = portfolio_values[start_index]
        end_value = portfolio_values[end_index]
        period_dates = dates[start_index : end_index + 1]
        period_values = portfolio_values[start_index : end_index + 1]

        # Calculate returns
        total_return = end_value - start_value
        total_return_pct = (total_return / start_value) * 100

        # Calculate daily returns for volatility (need at least 2 values)
        daily_returns = []
        for i in range(1, len(period_values)):
            daily_return_pct = (
                (period_values[i] - period_values[i - 1]) / period_values[i - 1]
            ) * 100
            daily_returns.append(daily_return_pct)

        # Key metrics - only calculate if we have sufficient data
        volatility = np.std(daily_returns) if len(daily_returns) >= 2 else 0
        best_day = max(daily_returns) if daily_returns else 0
        worst_day = min(daily_returns) if daily_returns else 0
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0

        # Calculate max drawdown (need at least 2 values)
        max_drawdown = 0
        if len(period_values) >= 2:
            peak = period_values[0]
            for value in period_values:
                peak = max(peak, value)
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)

        return {
            "period": period_name,
            "start_date": period_dates[0].strftime("%Y-%m-%d"),
            "end_date": period_dates[-1].strftime("%Y-%m-%d"),
            "days": len(period_dates),
            "starting_value": start_value,
            "ending_value": end_value,
            "total_return": total_return,
            "total_return_percent": total_return_pct,
            "average_daily_return": avg_daily_return,
            "volatility": volatility,
            "best_day_percent": best_day,
            "worst_day_percent": worst_day,
            "max_drawdown_percent": max_drawdown,
        }

    def find_period_start(days_back: int) -> Optional[int]:
        """Find the snapshot index that's approximately days_back from current"""
        target_date = current_date - timedelta(days=days_back)

        # Find the closest snapshot to the target date
        for i, snapshot_date in enumerate(dates):
            if snapshot_date >= target_date:
                return i
        return None  # Return None if no suitable found

    # Calculate metrics for different periods
    periods = {}

    # Start to End (full period) - always calculate if we have at least 2 snapshots
    if len(sorted_snapshots) >= 2:
        periods["full_period"] = calculate_period_metrics(
            0, len(sorted_snapshots) - 1, "Full Period"
        )

    # Past 1 Day (need at least 2 consecutive days)
    if len(sorted_snapshots) >= 2:
        one_day_metrics = calculate_period_metrics(
            len(sorted_snapshots) - 2, len(sorted_snapshots) - 1, "1 Day"
        )
        if one_day_metrics:
            periods["1_day"] = one_day_metrics

    # Past 1 Week (7 days) - need at least 2 data points in the period
    week_start_idx = find_period_start(7)
    if week_start_idx is not None and week_start_idx < len(sorted_snapshots) - 1:
        one_week_metrics = calculate_period_metrics(
            week_start_idx, len(sorted_snapshots) - 1, "1 Week"
        )
        if one_week_metrics and one_week_metrics["days"] >= 2:
            periods["1_week"] = one_week_metrics

    # Past 1 Month (30 days)
    month_start_idx = find_period_start(30)
    if month_start_idx is not None and month_start_idx < len(sorted_snapshots) - 1:
        one_month_metrics = calculate_period_metrics(
            month_start_idx, len(sorted_snapshots) - 1, "1 Month"
        )
        if one_month_metrics and one_month_metrics["days"] >= 2:
            periods["1_month"] = one_month_metrics

    # Past 3 Months (90 days)
    three_month_start_idx = find_period_start(90)
    if (
        three_month_start_idx is not None
        and three_month_start_idx < len(sorted_snapshots) - 1
    ):
        three_month_metrics = calculate_period_metrics(
            three_month_start_idx, len(sorted_snapshots) - 1, "3 Month"
        )
        if three_month_metrics and three_month_metrics["days"] >= 2:
            periods["3_month"] = three_month_metrics

    # Past 1 Year (365 days)
    year_start_idx = find_period_start(365)
    if year_start_idx is not None and year_start_idx < len(sorted_snapshots) - 1:
        one_year_metrics = calculate_period_metrics(
            year_start_idx, len(sorted_snapshots) - 1, "1 Year"
        )
        if one_year_metrics and one_year_metrics["days"] >= 2:
            periods["1_year"] = one_year_metrics

    # Current snapshot details
    current_snapshot = sorted_snapshots[-1]
    current_details = {
        "date": current_snapshot.date.strftime("%Y-%m-%d"),
        "portfolio_value": current_snapshot.portfolioValue,
        "positions_value": current_snapshot.positionsValue,
        "cash": current_snapshot.cash,
        "cash_percentage": (current_snapshot.cash / current_snapshot.portfolioValue)
        * 100,
    }

    # Overall summary
    overall_summary = {
        "analysis_date": current_date.strftime("%Y-%m-%d"),
        "total_days_analyzed": len(sorted_snapshots),
        "date_range": f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
        "current_portfolio_value": current_value,
        "available_periods": list(periods.keys()),
    }

    # Add full period metrics to summary if available
    if "full_period" in periods:
        full_period = periods["full_period"]
        if "total_return" in full_period:
            overall_summary["total_return_full_period"] = full_period["total_return"]
        if "total_return_percent" in full_period:
            overall_summary["total_return_percent_full_period"] = full_period[
                "total_return_percent"
            ]

    return {
        "overall_summary": overall_summary,
        "current_position": current_details,
        "period_performance": periods,  # Only includes periods with sufficient data
        "time_series_data": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "portfolio_values": portfolio_values,
            "daily_returns": [0]
            + [
                (
                    (portfolio_values[i] - portfolio_values[i - 1])
                    / portfolio_values[i - 1]
                )
                * 100
                for i in range(1, len(portfolio_values))
            ],
        },
    }


def create_performance_narrative(analysis: Dict[str, Any]) -> str:
    """
    Create a natural language narrative from the performance analysis
    Only includes periods that have sufficient data
    """
    if "error" in analysis:
        return f"Analysis error: {analysis['error']}"

    summary = analysis["overall_summary"]
    periods = analysis["period_performance"]
    current = analysis["current_position"]

    narrative = f"""
PORTFOLIO PERFORMANCE ANALYSIS
As of {summary["analysis_date"]}

CURRENT POSITION:
- Portfolio Value: ${current["portfolio_value"]:,.2f}
- Investment Value: ${current["positions_value"]:,.2f}
- Cash: ${current["cash"]:,.2f} ({current["cash_percentage"]:.1f}% of portfolio)

PERFORMANCE SUMMARY:
"""

    # Add performance for each available period
    period_order = ["1_day", "1_week", "1_month", "3_month", "1_year", "full_period"]
    period_names = {
        "1_day": "1 Day",
        "1_week": "1 Week",
        "1_month": "1 Month",
        "3_month": "3 Month",
        "1_year": "1 Year",
        "full_period": "Full Period",
    }

    available_periods_added = False
    for period_key in period_order:
        if period_key in periods:
            available_periods_added = True
            period = periods[period_key]
            arrow = "📈" if period["total_return_percent"] >= 0 else "📉"
            narrative += f"\n{period_names[period_key]} Performance {arrow}:"
            narrative += f"\n  Return: {period['total_return_percent']:+.2f}% (${period['total_return']:+,.2f})"
            narrative += f"\n  Volatility: {period['volatility']:.2f}%"

            # Only show best/worst day if we have multiple days
            if period["days"] > 1:
                narrative += f"\n  Best Day: {period['best_day_percent']:+.2f}%"
                narrative += f"\n  Worst Day: {period['worst_day_percent']:+.2f}%"
                narrative += f"\n  Max Drawdown: {period['max_drawdown_percent']:.2f}%"

            narrative += f"\n  Period: {period['start_date']} to {period['end_date']} ({period['days']} days)"

    if not available_periods_added:
        narrative += "\nNo sufficient data available for period analysis."

    # Add key insights only if we have multiple periods
    valid_periods = [p for p in periods.values() if p["days"] > 1]
    if len(valid_periods) >= 2:
        narrative += "\n\nKEY INSIGHTS:"

        best_period = max(valid_periods, key=lambda x: x["total_return_percent"])
        worst_period = min(valid_periods, key=lambda x: x["total_return_percent"])

        narrative += f"\n- Strongest period: {best_period['period']} ({best_period['total_return_percent']:+.2f}%)"
        narrative += f"\n- Weakest period: {worst_period['period']} ({worst_period['total_return_percent']:+.2f}%)"

        # Risk assessment
        if "full_period" in periods:
            volatility = periods["full_period"]["volatility"]
            if volatility < 2:
                risk_level = "Low"
            elif volatility < 5:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            narrative += f"\n- Risk Level: {risk_level} (Volatility: {volatility:.2f}%)"

    narrative += f"\n\nData Coverage: {summary['date_range']} ({summary['total_days_analyzed']} days of data)"

    return narrative
