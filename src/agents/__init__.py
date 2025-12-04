from prisma.enums import Role
from src.prompt.roles import ROLE_PROMPTS_MAP
from src.agents.chief_investment_officer.agent import (
    build_chief_investment_officer_agent,
)
from src.agents.market_analyst.agent import build_market_analyst_agent
from src.agents.fundamental_analyst.agent import build_fundamental_analyst_agent
from src.agents.risk_analyst.agent import build_risk_analyst_agent
from src.agents.equity_research_analyst.agent import build_equity_research_analyst_agent


registered_agents = [
    Role.MARKET_ANALYST,
    Role.RISK_ANALYST,
    Role.EQUITY_RESEARCH_ANALYST,
    Role.FUNDAMENTAL_ANALYST,
]


REGISTERED_AGENTS: dict[Role, str] = {}

for role in registered_agents:
    REGISTERED_AGENTS[role] = ROLE_PROMPTS_MAP[role]


ORCHESTRATOR_AGENT = Role.CHIEF_INVESTMENT_OFFICER

__all__ = [
    "REGISTERED_AGENTS",
    "build_chief_investment_officer_agent",
    "build_market_analyst_agent",
    "build_fundamental_analyst_agent",
    "build_risk_analyst_agent",
    "build_equity_research_analyst_agent",
]
