from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_technical_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.TECHNICAL_ANALYST)
    langchain_model = get_model(context.llm_model)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.execute_python_technical_analysis,
            tools.download_ticker_bars_data,
            tools.get_recommend_stock_tool(Role.TECHNICAL_ANALYST),
        ],
        middleware=[
            middleware.LoggingMiddleware(Role.TECHNICAL_ANALYST.value),
            middleware.CleanUpPythonMiddleware(),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
