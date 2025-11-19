from collections import defaultdict
from datetime import timedelta, date, datetime
from typing import Dict, List, Optional, TypedDict
import numpy as np
from src.services.sandx_ai.typing import TimelineValue


class FormattedTimelineValue(TypedDict):
    date: date
    value: float


class PeriodMetrics(TypedDict):
    period: str
    start_date: str
    end_date: str
    days: int
    starting_value: float
    ending_value: float
    total_return: float
    total_return_percent: float
    average_daily_return: float
    volatility: float
    best_day_percent: float
    worst_day_percent: float
    max_drawdown_percent: float


class CurrentDetails(TypedDict):
    date: str
    portfolio_value: float
    positions_value: float
    cash: float
    cash_percentage: float


class AnalysisSummary(TypedDict):
    analysis_date: str
    total_days_analyzed: int
    date_range: str
    current_portfolio_value: float
    available_periods: List[str]
    total_return_full_period: float
    total_return_percent_full_period: float


class TimeSeriesData(TypedDict):
    dates: List[str]
    portfolio_values: List[float]
    daily_returns: List[float]


class AnalysisResult(TypedDict):
    overall_summary: AnalysisSummary
    period_performance: Dict[str, PeriodMetrics]
    time_series_data: TimeSeriesData


def deduplicate_timeline_by_date(timeline_values: List[TimelineValue]):
    """
    Groups timeline entries by calendar date and keeps only the latest entry per date.

    Args:
        timeline_values (list): List of dicts with keys 'date' (ISO string) and 'value'.

    Returns:
        list: Deduplicated list with one entry per calendar date, converted to date objects.
    """
    grouped: Dict[str, List[TimelineValue]] = defaultdict(list)
    for entry in timeline_values:
        date_key = entry["date"][:10]  # Extract YYYY-MM-DD
        grouped[date_key].append(entry)

    def convert_to_date(entry: TimelineValue) -> FormattedTimelineValue:
        return {
            "date": datetime.fromisoformat(entry["date"].replace("Z", "+00:00")).date(),
            "value": entry["value"],
        }

    transformed_timeline_values = [
        convert_to_date(
            max(
                group,
                key=lambda x: datetime.fromisoformat(x["date"].replace("Z", "+00:00")),
            )
        )
        for group in grouped.values()
    ]
    return transformed_timeline_values


def compute_daily_returns(values_slice: List[float]) -> np.ndarray:
    if len(values_slice) < 2:
        return np.array([], dtype=float)
    arr = np.array(values_slice, dtype=float)
    prev = arr[:-1]
    curr = arr[1:]
    return ((curr - prev) / prev) * 100


def compute_max_drawdown(values_slice: List[float]) -> float:
    if len(values_slice) < 2:
        return 0.0
    arr = np.array(values_slice, dtype=float)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (peaks - arr) / peaks * 100
    return float(np.max(drawdowns)) if drawdowns.size >= 2 else 0.0


def find_period_start_index(
    dates: List[date],  # asc ordering
    current_date: date,
    days_back: int,
) -> Optional[int]:
    target_date = current_date - timedelta(days=days_back)
    for i, snapshot_date in enumerate(dates):
        if snapshot_date >= target_date:
            return i
    return None


def period_metrics(  # pylint: disable=too-many-arguments
    dates: List[date],
    values: List[float],
    start_index: int,
    end_index: int,
    period_name: str,
) -> Optional[PeriodMetrics]:
    if start_index < 0 or start_index >= end_index:
        return None
    if end_index - start_index < 1:
        return None

    start_value = values[start_index]
    end_value = values[end_index]
    values_slice = values[start_index : end_index + 1]

    total_return = end_value - start_value
    total_return_pct = (total_return / start_value) * 100

    daily_returns = compute_daily_returns(values_slice)
    volatility = float(np.std(daily_returns)) if daily_returns.size >= 2 else 0.0
    best_day = float(np.max(daily_returns)) if daily_returns.size else 0.0
    worst_day = float(np.min(daily_returns)) if daily_returns.size else 0.0
    avg_daily_return = float(np.mean(daily_returns)) if daily_returns.size else 0.0
    max_drawdown = compute_max_drawdown(values_slice)

    return PeriodMetrics(
        period=period_name,
        start_date=dates[start_index].strftime("%Y-%m-%d"),
        end_date=dates[end_index].strftime("%Y-%m-%d"),
        days=end_index - start_index + 1,
        starting_value=start_value,
        ending_value=end_value,
        total_return=total_return,
        total_return_percent=total_return_pct,
        average_daily_return=avg_daily_return,
        volatility=volatility,
        best_day_percent=best_day,
        worst_day_percent=worst_day,
        max_drawdown_percent=max_drawdown,
    )


def build_periods(
    dates: List[date],  # asc ordering
    values: List[float],
    current_date: date,
    last_index: int,
) -> Dict[str, PeriodMetrics]:
    periods: Dict[str, PeriodMetrics] = {}

    full = period_metrics(dates, values, 0, last_index, "Full Period")
    if full:
        periods["full_period"] = full

    one_day = period_metrics(dates, values, max(0, last_index - 1), last_index, "1 Day")
    if one_day:
        periods["1_day"] = one_day

    specs = [
        ("1_week", 7, "1 Week"),
        ("1_month", 30, "1 Month"),
        ("3_month", 90, "3 Month"),
        ("1_year", 365, "1 Year"),
    ]
    for key, days_back, label in specs:
        start_idx = find_period_start_index(dates, current_date, days_back)
        if start_idx is None or start_idx >= last_index:
            continue
        m = period_metrics(dates, values, start_idx, last_index, label)
        if m and m["days"] >= 2:
            periods[key] = m
    return periods


def build_overall_summary(
    dates: List[date],
    current_date: date,
    current_value: float,
    periods: Dict[str, PeriodMetrics],
) -> AnalysisSummary:
    summary: AnalysisSummary = {
        "analysis_date": current_date.strftime("%Y-%m-%d"),
        "total_days_analyzed": len(dates),
        "date_range": f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
        "current_portfolio_value": current_value,
        "available_periods": list(periods.keys()),
        "total_return_full_period": 0.0,
        "total_return_percent_full_period": 0.0,
    }
    if "full_period" in periods:
        fp = periods["full_period"]
        summary["total_return_full_period"] = fp["total_return"]
        summary["total_return_percent_full_period"] = fp["total_return_percent"]
    return summary


def build_time_series_data(
    dates: List[date], portfolio_values: List[float]
) -> TimeSeriesData:
    dates = dates[-365:]
    portfolio_values = portfolio_values[-365:]

    daily = [0.0] + [
        ((portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1])
        * 100
        for i in range(1, len(portfolio_values))
    ]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "portfolio_values": portfolio_values,
        "daily_returns": daily,
    }


def analyze_timeline_value(
    timeline_values: List[TimelineValue],
) -> Optional[AnalysisResult]:
    if len(timeline_values) < 2:
        return None

    snapshots = deduplicate_timeline_by_date(timeline_values)

    sorted_snapshots = sorted(snapshots, key=lambda x: x["date"])
    dates_local = [s["date"] for s in sorted_snapshots]
    values_local = [s["value"] for s in sorted_snapshots]

    current_date_local = sorted_snapshots[-1]["date"]
    current_value_local = values_local[-1]
    last_index_local = len(sorted_snapshots) - 1

    periods_local = build_periods(
        dates_local, values_local, current_date_local, last_index_local
    )
    overall_summary_local = build_overall_summary(
        dates_local, current_date_local, current_value_local, periods_local
    )
    time_series_data_local = build_time_series_data(dates_local, values_local)

    return {
        "overall_summary": overall_summary_local,
        "period_performance": periods_local,
        "time_series_data": time_series_data_local,
    }


__all__ = ["analyze_timeline_value", "create_performance_narrative"]


def create_performance_narrative(analysis: AnalysisResult) -> str:
    """
    Create a natural language narrative from the performance analysis
    Only includes periods that have sufficient data
    """

    summary = analysis["overall_summary"]
    periods = analysis["period_performance"]

    narrative = f"""
User's PORTFOLIO PERFORMANCE ANALYSIS
As of {summary["analysis_date"]}

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
