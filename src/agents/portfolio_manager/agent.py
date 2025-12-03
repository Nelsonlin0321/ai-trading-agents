# src/agents/portfolio_manager/agent.py
from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.typings import ModelName
from src.context import build_context, Context
from src.prompt import build_agent_system_prompt


async def build_portfolio_manager_agent(model_name: ModelName, run_id: str):
    context = await build_context(run_id)
    system_prompt = await build_agent_system_prompt(context, Role.PORTFOLIO_MANAGER)
    langchain_model = get_model(model_name)

    # Add trading tools
    agent = create_agent(
        model=langchain_model,
        tools=[
            # Analysis tools
            tools.get_fundamental_data,
            tools.get_latest_market_news,
            tools.get_stock_live_historical_price_change,
            tools.get_portfolio_performance_analysis,
            tools.list_current_positions,
            tools.get_user_investment_strategy,
            # Trading tools (NEW)
            tools.buy_stock,
            tools.sell_stock,
            # tools.get_latest_quotes,
            # tools.get_market_hours,   # Need to add this
        ],
        middleware=[
            middleware.summarization_middleware,  # type: ignore
            middleware.todo_list_middleware,
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )
    return context, agent
