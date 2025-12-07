from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_market_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.MARKET_ANALYST)
    langchain_model = get_model(context.model_name)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_latest_market_news,
            tools.do_google_market_research,
            tools.get_etf_live_historical_price_change,
            tools.get_stock_live_historical_price_change,
            tools.list_current_positions,
            tools.get_most_active_stockers,
            tools.get_portfolio_performance_analysis,
            tools.get_user_investment_strategy,
        ],
        middleware=[
            middleware.summarization_middleware,  # type: ignore
            middleware.todo_list_middleware,
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
