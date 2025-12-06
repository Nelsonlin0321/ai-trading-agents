from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src.models import get_model
from src.typings import ModelName
from src.context import build_context, Context
from src.prompt import build_agent_system_prompt


async def build_chief_investment_officer_agent(model_name: ModelName, run_id: str):
    """
    Build the Chief Investment Officer agent with comprehensive orchestration capabilities.

    The CIO agent can:
    1. Access all analytical tools across different domains
    2. Coordinate between different specialist agents
    3. Make final investment decisions
    4. Monitor overall portfolio risk and performance
    5. Provide strategic direction
    """
    context = await build_context(run_id)
    system_prompt = await build_agent_system_prompt(
        context, Role.CHIEF_INVESTMENT_OFFICER
    )
    langchain_model = get_model(model_name)

    agent = create_agent(
        model=langchain_model,
        tools=[
            # Portfolio management tools
            tools.list_current_positions,
            tools.get_portfolio_performance_analysis,
            tools.get_user_investment_strategy,
            tools.buy_stock,
            tools.sell_stock,
            tools.get_latest_quotes,
            tools.get_latest_quote,
        ],
        # middleware=[
        #     middleware.summarization_middleware,  # type: ignore
        #     middleware.todo_list_middleware,
        # ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return context, agent
