from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_risk_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.RISK_ANALYST)
    langchain_model = get_model(context.model_name)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_fundamental_risk_data,
            tools.get_price_risk_indicators,
            tools.get_volatility_risk_indicators,
            tools.get_stock_live_historical_price_change,
            tools.get_recommend_stock_tool(Role.RISK_ANALYST),
        ],
        middleware=[
            middleware.summarization_middleware,  # type: ignore
            middleware.todo_list_middleware,
            middleware.LoggingMiddleware(Role.RISK_ANALYST.value),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
