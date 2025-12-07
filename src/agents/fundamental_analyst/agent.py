from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_fundamental_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.FUNDAMENTAL_ANALYST)
    langchain_model = get_model(context.model_name)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_fundamental_data,
            tools.get_stock_live_historical_price_change,
        ],
        middleware=[
            middleware.summarization_middleware,  # type: ignore
            middleware.todo_list_middleware,
            middleware.LoggingMiddleware("FUNDAMENTAL_ANALYST"),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
