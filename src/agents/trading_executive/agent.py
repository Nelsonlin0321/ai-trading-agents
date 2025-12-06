from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src.models import get_model
from src.typings import ModelName
from src.context import build_context, Context
from src.prompt import build_agent_system_prompt


async def build_trading_executor_agent(model_name: ModelName, run_id: str):
    context = await build_context(run_id)
    system_prompt = await build_agent_system_prompt(context, Role.TRADING_EXECUTOR)
    langchain_model = get_model(model_name)

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
        # middleware=[
        #     middleware.summarization_middleware,  # type: ignore
        #     middleware.todo_list_middleware,
        # ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return context, agent
