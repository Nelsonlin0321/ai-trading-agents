from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_trading_executor_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.TRADING_EXECUTOR)
    langchain_model = get_model(context.model_name)

    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.list_current_positions,
            tools.get_user_investment_strategy,
            tools.buy_stock,
            tools.sell_stock,
            tools.get_latest_quotes,
            tools.get_latest_quote,
            tools.get_market_status,
        ],
        middleware=[
            middleware.LoggingMiddleware("TRADING_EXECUTOR"),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
