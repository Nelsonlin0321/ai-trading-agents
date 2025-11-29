from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.typings import ModelName
from src.context import build_context, Context
from src.prompt import build_agent_system_prompt


async def build_risk_analyst_agent(model_name: ModelName, run_id: str):
    context = await build_context(run_id)
    system_prompt = await build_agent_system_prompt(context, Role.RISK_ANALYST)
    langchain_model = get_model(model_name)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_fundamental_risk_data,
            tools.get_stock_live_historical_price_change,
        ],
        middleware=[
            middleware.summarization_middleware,  # type: ignore
            middleware.todo_list_middleware,
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return context, agent
