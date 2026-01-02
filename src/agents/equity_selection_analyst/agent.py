from langchain.agents import create_agent
from prisma.enums import Role
from src import middleware
from src import tools
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_equity_selection_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(
        context, Role.EQUITY_SELECTION_ANALYST
    )
    langchain_model = get_model(context.llm_model)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_user_investment_strategy,
            tools.get_selected_tickers,
            tools.write_down_selected_tickers,
            tools.list_current_positions,
            tools.get_historical_reviewed_tickers,
            tools.get_market_deep_research_analysis,
            tools.get_latest_market_news,
            tools.get_etf_live_historical_price_change,
            tools.get_stock_live_historical_price_change,
            tools.get_most_active_stockers,
        ],
        middleware=[
            # middleware.summarization_middleware,  # type: ignore
            # middleware.todo_list_middleware,
            middleware.LoggingMiddleware(Role.EQUITY_SELECTION_ANALYST.value),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
