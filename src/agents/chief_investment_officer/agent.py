from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src.tools import handoff_tools
from src.middleware import (
    LoggingMiddleware,
    todo_list_middleware,
    # summarization_middleware,
)
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_chief_investment_officer_agent(context: Context):
    system_prompt = await build_agent_system_prompt(
        context, Role.CHIEF_INVESTMENT_OFFICER
    )

    langchain_model = get_model(context.llm_model)

    agent = create_agent(
        model=langchain_model,
        tools=[
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
            handoff_tools.handoff_to_equity_research_analyst,
            handoff_tools.handoff_to_fundamental_analyst,
            handoff_tools.handoff_to_technical_analyst,
            handoff_tools.handoff_to_risk_analyst,
            handoff_tools.handoff_to_trading_executor,
        ],
        middleware=[
            todo_list_middleware,  # type: ignore
            # summarization_middleware,
            LoggingMiddleware("CHIEF_INVESTMENT_OFFICER"),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
