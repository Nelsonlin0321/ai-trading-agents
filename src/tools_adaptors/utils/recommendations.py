from prisma.models import Recommend


def format_recommendations_markdown(recommendations: list[Recommend]) -> str:
    """Format recommendations as a markdown table.

    Args:
        recommendations: List of recommendations

    Returns:
        A markdown-formatted table with the recommendations.
    """
    if not recommendations:
        return "No recommendations found."

    headers = [
        "Ticker",
        "Type",
        "Amount",
        "Limit Price",
        "Allocation",
        "Confidence",
        "Analyst",
        "Rationale",
    ]
    rows = [
        "| "
        + " | ".join(
            [
                rec.ticker,
                rec.type.value if hasattr(rec.type, "value") else str(rec.type),
                str(rec.amount),
                f"{rec.limitPrice:.2f}" if rec.limitPrice else "None",
                f"{rec.allocation:.2%}",
                f"{rec.confidence:.2%}",
                rec.role,
                rec.rationale.replace("\n", " ").replace("|", "\\|"),
            ]
        )
        + " |"
        for rec in recommendations
    ]

    header_row = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"

    return "\n".join([header_row, separator] + rows)


__all__ = ["format_recommendations_markdown"]
