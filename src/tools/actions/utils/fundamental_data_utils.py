from src.utils.constants import FUNDAMENTAL_CATEGORIES


def preprocess_info_dict(info_dict):
    if "dividendYield" in info_dict:
        info_dict["dividendYield"] = float(info_dict["dividendYield"]) / 100
    if "fiveYearAvgDividendYield" in info_dict:
        info_dict["fiveYearAvgDividendYield"] = (
            float(info_dict["fiveYearAvgDividendYield"]) / 100
        )
    return info_dict


def categorize_fundamental_data(info_dict):
    """
    Categorize fundamental data from info dictionary into structured categories

    Args:
        info_dict (dict): The yfinance info dictionary

    Returns:
        dict: Structured data with categories containing available metrics
    """
    categorized_data = {}

    for category, metric_keys in FUNDAMENTAL_CATEGORIES.items():
        category_metrics = {}

        for key in metric_keys:
            if key in info_dict:
                category_metrics[key] = info_dict[key]

        # Only include categories that have data
        if category_metrics:
            categorized_data[category] = category_metrics

    return categorized_data


def get_categorized_metrics(info_dict, format_values=True, include_empty=False):
    """
    Get categorized metrics with advanced options

    Args:
        info_dict (dict): yfinance info dictionary
        categories_map (dict): Category mapping
        format_values (bool): Whether to format values for display
        include_empty (bool): Whether to include empty categories

    Returns:
        dict: Categorized metrics data
    """

    def format_metric_value(key, value):
        """Helper function to format metric values"""
        if value is None:
            return "N/A"

        # Large monetary values
        if isinstance(value, (int, float)) and abs(value) >= 1e9:
            return f"{value / 1e9:.2f}B"
        elif isinstance(value, (int, float)) and abs(value) >= 1e6:
            return f"{value / 1e6:.2f}M"
        elif isinstance(value, (int, float)) and abs(value) >= 1e3:
            return f"{value / 1e3:.2f}K"

        # Percentages for ratios, margins, yields, returns
        percent_indicators = ["margin", "yield", "return", "growth"]
        if isinstance(value, float) and any(
            indicator in key.lower() for indicator in percent_indicators
        ):
            return f"{value:.2%}"

        # Simple floats
        elif isinstance(value, float):
            return f"{value:.4f}"

        # Integers
        elif isinstance(value, int):
            return f"{value:,}"

        return str(value)

    categorized_data = {}

    for category, metric_keys in FUNDAMENTAL_CATEGORIES.items():
        category_metrics = {}

        for key in metric_keys:
            if key in info_dict:
                value = info_dict[key]
                if format_values:
                    category_metrics[key] = format_metric_value(key, value)
                else:
                    category_metrics[key] = value

        # Decide whether to include empty categories
        if category_metrics or include_empty:
            categorized_data[category] = category_metrics

    return categorized_data


def format_fundamentals_markdown(categorized_data, ticker_symbol):
    """
    Simple markdown format with tables for each category
    """
    md = [f"# {ticker_symbol} Fundamental Data", ""]

    for category, metrics in categorized_data.items():
        if metrics:
            md.append(f"## {category}")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            for key, value in metrics.items():
                md.append(f"| {key} | {value} |")
            md.append("")

    return "\n".join(md)


__all__ = [
    "categorize_fundamental_data",
    "get_categorized_metrics",
    "format_fundamentals_markdown",
    "preprocess_info_dict",
]
