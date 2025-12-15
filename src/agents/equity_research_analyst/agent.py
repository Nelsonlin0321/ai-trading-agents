from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_equity_research_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(
        context, Role.EQUITY_RESEARCH_ANALYST
    )
    langchain_model = get_model(context.llm_model)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.do_google_equity_research,
            tools.get_latest_equity_news,
            tools.get_price_trend,
            tools.get_recommend_stock_tool(Role.EQUITY_RESEARCH_ANALYST),
        ],
        middleware=[
            # middleware.summarization_middleware,  # type: ignore
            # middleware.todo_list_middleware,
            middleware.LoggingMiddleware(Role.EQUITY_RESEARCH_ANALYST.value),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
