from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src.utils.constants import ALL_ROLES, OPTIONAL_ROLES
from src.tools import handoff_tools
from src.middleware import (
    LoggingMiddleware,
    todo_list_middleware,
    # summarization_middleware,
)
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt, AGENT_DESCRIPTIONS

agent_handoff_tool_map = {
    "EQUITY_RESEARCH_ANALYST": handoff_tools.handoff_to_equity_research_analyst,
    "FUNDAMENTAL_ANALYST": handoff_tools.handoff_to_fundamental_analyst,
    "TECHNICAL_ANALYST": handoff_tools.handoff_to_technical_analyst,
    "RISK_ANALYST": handoff_tools.handoff_to_risk_analyst,
}


async def build_chief_investment_officer_agent(context: Context):
    system_prompt = await build_agent_system_prompt(
        context, Role.CHIEF_INVESTMENT_OFFICER
    )

    agents = context.bot.agents.split(",") if context.bot.agents else ALL_ROLES

    optional_handoff_tools = []
    for role in OPTIONAL_ROLES:
        if role in agents:
            optional_handoff_tools.append(agent_handoff_tool_map[role])

    team_description: str = "\n\n ## YOUR INVESTMENT TEAM:\n\n" + "\n".join(
        f"### {idx}. {AGENT_DESCRIPTIONS[role]['title']}\n"
        f"**Description:** {AGENT_DESCRIPTIONS[role]['description']}\n"
        f"**Capabilities:** {AGENT_DESCRIPTIONS[role]['key_capabilities']}\n"
        # f"**Default Strength weight in decisions:** {int(AGENT_DESCRIPTIONS[role]['strength_weight'] * 100)}%\n"
        for idx, role in enumerate(AGENT_DESCRIPTIONS.keys(), start=1)
        if role in agents
    )

    system_prompt = system_prompt + team_description

    langchain_model = get_model(context.llm_model)

    tool_list = [
        tools.get_portfolio_performance_analysis,
        tools.get_user_investment_strategy,
        tools.get_analysts_recommendations,
        tools.get_recommend_stock_tool(Role.CHIEF_INVESTMENT_OFFICER),
        tools.send_summary_email_tool,
        tools.write_summary_report,
        tools.get_selected_tickers,
        tools.get_market_status,
        tools.take_learning_note,
        tools.get_learning_notes,
        handoff_tools.handoff_to_market_analyst,
        handoff_tools.handoff_to_equity_selection_analyst,
        handoff_tools.handoff_to_trading_executor,
    ] + optional_handoff_tools

    agent = create_agent(
        model=langchain_model,
        tools=tool_list,
        middleware=[
            todo_list_middleware,  # type: ignore
            # summarization_middleware,
            LoggingMiddleware("CHIEF_INVESTMENT_OFFICER"),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
