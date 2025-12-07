from prisma.enums import Role
from src.agents.market_analyst.agent import build_market_analyst_agent
from src.agents.fundamental_analyst.agent import build_fundamental_analyst_agent
from src.agents.risk_analyst.agent import build_risk_analyst_agent
from src.agents.equity_research_analyst.agent import build_equity_research_analyst_agent
from src.agents.trading_executive.agent import build_trading_executor_agent


ORCHESTRATOR_AGENT = Role.CHIEF_INVESTMENT_OFFICER

__all__ = [
    "build_trading_executor_agent",
    "build_market_analyst_agent",
    "build_fundamental_analyst_agent",
    "build_risk_analyst_agent",
    "build_equity_research_analyst_agent",
]
