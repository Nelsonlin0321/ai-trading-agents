from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.typings import ModelName
from src.context import get_context, Context, build_context_narrative


async def get_us_market_analyst_agent(model_name: ModelName, role: Role, run_id: str):
    context = await get_context(run_id)
    system_prompt = await build_context_narrative(context, role)
    langchain_model = get_model(model_name)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_latest_market_news,
            tools.google_market_research,
            tools.etf_live_historical_price_change,
            tools.stock_live_historical_price_change,
            tools.list_current_positions,
            tools.most_active_stockers,
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
