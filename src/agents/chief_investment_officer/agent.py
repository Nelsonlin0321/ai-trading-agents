from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src.tools.handoff_tools import handoff_to_specialist
from src.middleware import (
    LoggingMiddleware,
    todo_list_middleware,
    summarization_middleware,
)
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_chief_investment_officer_agent(context: Context):
    system_prompt = await build_agent_system_prompt(
        context, Role.CHIEF_INVESTMENT_OFFICER
    )
    langchain_model = get_model(context.model_name)

    agent = create_agent(
        model=langchain_model,
        tools=[
            # Portfolio management tools
            tools.list_current_positions,
            tools.get_portfolio_performance_analysis,
            tools.get_user_investment_strategy,
            # tools.buy_stock,
            # tools.sell_stock,
            tools.get_latest_quotes,
            tools.get_latest_quote,
            tools.get_analysts_recommendations,
            handoff_to_specialist,
        ],
        middleware=[
            todo_list_middleware,  # type: ignore
            summarization_middleware,
            LoggingMiddleware("CHIEF_INVESTMENT_OFFICER"),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
