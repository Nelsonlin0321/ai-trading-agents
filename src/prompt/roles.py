from typing import TypedDict
from prisma.enums import Role
from src.utils import read_text
from src.typings.agent_roles import SubAgentRole


class AgentDescription(TypedDict):
    title: str
    description: str
    key_capabilities: list[str]
    # strength_weight: float


# Agent descriptions for reference in the CIO prompt
AGENT_DESCRIPTIONS: dict[SubAgentRole, AgentDescription] = {
    "MARKET_ANALYST": {
        "title": "Market Analyst",
        "description": "Senior US-market analyst who delivers concise, actionable briefings on market catalysts, drivers, and sentiment inflections.",
        "key_capabilities": [
            "Overnight and breaking headline catalysts",
            "Key macro, sector, and single-stock drivers",
            "Imminent event risk (earnings, Fed speakers, data releases)",
            "Cross-asset flow and sentiment inflections",
        ],
        # "strength_weight": 0.25
    },
    "EQUITY_SELECTION_ANALYST": {
        "title": "Equity Selection Analyst",
        "description": "An expert that intelligently selects tickers for deeper investment analysis by translating deep market research insights into actionable equity choices, while strictly respecting portfolio constraints.",
        "key_capabilities": [
            "Reviews comprehensive market research to identify the most relevant investment themes, trends, and catalysts",
            "Selects exactly 2 existing holdings for reassessment based on new material impacts from research",
            "Identifies 2 high-conviction new tickers that offer fresh exposure or alpha potential",
        ],
        # "strength_weight": 0.2,  # Equity selection is important
    },
    "RISK_ANALYST": {
        "title": "Risk Analyst",
        "description": "Meticulous risk analyst who quantifies downside scenarios, stress-tests portfolios, and designs hedging frameworks.",
        "key_capabilities": [
            "Quantifies downside scenarios and potential losses",
            "Stress-tests portfolios under various market conditions",
            "Designs hedging frameworks for risk mitigation",
        ],
        # "strength_weight": 0.2,  # Risk management is equally important
    },
    "EQUITY_RESEARCH_ANALYST": {
        "title": "Equity Research Analyst",
        "description": "Senior equity research analyst focused on catalysts, drivers, and market inflections for specific securities.",
        "key_capabilities": [
            "Catalyst-driven analysis of specific equities",
            "Sector and thematic research",
            "Event risk assessment for individual stocks",
            "Market sentiment and flow analysis for securities",
        ],
        # "strength_weight": 0.5,  # Catalysts and timing are important
    },
    "FUNDAMENTAL_ANALYST": {
        "title": "Fundamental Analyst",
        "description": "Fundamental equity analyst who builds conviction from first principles using comprehensive financial metrics.",
        "key_capabilities": [
            "Data sanity checks and anomaly detection",
            "Extracts 5-7 high-signal insights with metric labels and values",
            "Assesses quality, durability, margins, returns, cash conversion",
            "Evaluates valuation vs growth using DCF/comps",
            "Analyzes capital returns and payout sustainability",
            "Outlines key catalysts and risks with monitoring indicators",
        ],
        # "strength_weight": 0.30,  # Fundamentals provide the valuation anchor
    },
    "TECHNICAL_ANALYST": {
        "title": "Technical Analyst",
        "description": "Technical analyst who performs technical analysis on ticker data using Python code execution.",
        "key_capabilities": [
            "Perform technical analysis on historical price data",
            "Execute Python code to calculate indicators and print out the results",
            "Provide buy/sell signals based on technical indicators",
        ],
        # "strength_weight": 0.20,  # Technical analysis provides timing and trend confirmation
    },
    "TRADING_EXECUTOR": {
        "title": "Trading Executor",
        "description": "Trading executor who executes trades based on instructions from the Chief Investment Officer.",
        "key_capabilities": [
            "Verify watchlist/position, market hours, cash, holdings",
            "Execute BUY/SELL exactly as instructed",
            "Confirm trade booked, cash/position updated",
            "Ensure cash sufficiency for buys",
            "Never short-sell (≤ current holdings)",
        ],
        # "strength_weight": 0,  # Execution role, lower weight in decision synthesis
    },
}

RECOMMENDATION_PROMPT: str = "\n\n In addition to analysis, based on your analysis, you should frame your final recommendation and state BUY, SELL, or HOLD with your rationale, Allocation Percentage, and confidence level (0.0-1.0)"


AGENT_TEAM_DESCRIPTION: str = "## YOUR INVESTMENT TEAM:\n\n" + "\n".join(
    f"### {idx}. {AGENT_DESCRIPTIONS[role]['title']}\n"
    f"**Description:** {AGENT_DESCRIPTIONS[role]['description']}\n"
    f"**Capabilities:** {AGENT_DESCRIPTIONS[role]['key_capabilities']}\n"
    # f"**Default Strength weight in decisions:** {int(AGENT_DESCRIPTIONS[role]['strength_weight'] * 100)}%\n"
    for idx, role in enumerate(AGENT_DESCRIPTIONS.keys(), start=1)
)

CHIEF_INVESTMENT_OFFICER_ROLE_PROMPT: str = (
    read_text("src/prompt/chief_investment_officer.md") + AGENT_TEAM_DESCRIPTION
)


RolePromptMap = dict[Role, str]

ROLE_PROMPTS_MAP: RolePromptMap = {
    Role.MARKET_ANALYST: read_text("src/prompt/market_analyst.md"),
    Role.EQUITY_SELECTION_ANALYST: read_text("src/prompt/equity_selection_analyst.md"),
    Role.EQUITY_RESEARCH_ANALYST: read_text("src/prompt/equity_research_analyst.md"),
    Role.CHIEF_INVESTMENT_OFFICER: CHIEF_INVESTMENT_OFFICER_ROLE_PROMPT,
    Role.RISK_ANALYST: read_text("src/prompt/risk_analyst.md") + RECOMMENDATION_PROMPT,
    Role.FUNDAMENTAL_ANALYST: read_text("src/prompt/fundamental_analyst.md")
    + RECOMMENDATION_PROMPT,
    Role.TECHNICAL_ANALYST: read_text("src/prompt/technical_analyst.md")
    + RECOMMENDATION_PROMPT,
    Role.TRADING_EXECUTOR: read_text("src/prompt/trading_executor.md"),
}


__all__ = ["RECOMMENDATION_PROMPT", "ROLE_PROMPTS_MAP", "AGENT_TEAM_DESCRIPTION"]
