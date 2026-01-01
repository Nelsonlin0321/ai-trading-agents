import json
from loguru import logger
from datetime import datetime, timezone
from langchain_core.messages import AnyMessage
from langgraph.runtime import Runtime
from langchain.agents import middleware
from prisma.types import (
    AgentMessageCreateInput,
)
from src import db
from src.db import CachedAgentMessage, CACHED_AGENTS_MESSAGES
from src.typings.context import Context
from src.models import get_model
from src.utils.constants import ALL_ROLES
from src.typings.agent_roles import AgentRole
from src.utils import async_retry

PERSISTED_MSG_IDS: set[str] = set()

langchain_model = get_model("minimax/minimax-m2.1")


summarization_middleware = middleware.SummarizationMiddleware(
    model=langchain_model,
    max_tokens_before_summary=128_000,
    messages_to_keep=20,
)

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

## Example Planning for Regular Task
Here is an example of how to plan a regular investment task:
1) Review user's investment strategy and portfolio performance
2) Get market analysis from Market Analyst
3) Delegate Ticker Prioritization to Equity Selection Analyst to select 4 tickers, 2 of existing tickers, 2 of new tickers for deep dive analysis if the user didn't specify any tickers
5) For each selected ticker, conduct parallel analysis on with Equity, Fundamental Analysts,Technical Analyst, and then finally Risk Analyst
6) Synthesize analysis results into final recommendations
7) Execute trades by handoff to trading executor if market is open and recommendations are high-confidence
8) Compile and send final investment report"""  # noqa: E501

todo_list_middleware = middleware.TodoListMiddleware(
    system_prompt=WRITE_TODOS_SYSTEM_PROMPT
)


class LoggingMiddleware(middleware.AgentMiddleware[middleware.AgentState, Context]):
    def __init__(self, role: AgentRole):
        assert role in ALL_ROLES
        self.role: AgentRole = role

    #  Only one agent called this middleware at a time
    async def aafter_model(
        self, state: middleware.AgentState, runtime: Runtime[Context]
    ) -> None:
        context = runtime.context
        messages = state["messages"]
        # Fire-and-forget: schedule persistence in the background without awaiting

        await persist_agent_messages(
            role=self.role,
            bot_id=context.bot.id,
            run_id=context.run.id,
            messages=messages,
        )

        await cache_agent_messages(
            role=self.role,
            bot_id=context.bot.id,
            run_id=context.run.id,
            messages=messages,
        )


@async_retry()
async def persist_agent_message(
    role: AgentRole, bot_id: str, run_id: str, message: AnyMessage
):
    if not message.id:
        return

    if message.id in PERSISTED_MSG_IDS:
        return

    existed_message = await db.prisma.agentmessage.find_unique(
        where={"id": message.id},
    )

    if not existed_message:
        await db.prisma.agentmessage.create(
            data=AgentMessageCreateInput(
                id=message.id,
                role=role,
                botId=bot_id,
                messages=message.model_dump_json(),
                runId=run_id,
            )
        )
    PERSISTED_MSG_IDS.add(message.id)


async def persist_agent_messages(
    role: AgentRole, bot_id: str, run_id: str, messages: list[AnyMessage]
):
    for msg in messages:
        await persist_agent_message(
            role=role,
            bot_id=bot_id,
            run_id=run_id,
            message=msg,
        )


@async_retry()
async def cache_agent_messages(
    role: AgentRole, bot_id: str, run_id: str, messages: list[AnyMessage]
):
    # Save and consolidated agent messages to local file because cross agent with different message states

    existing_msg_ids = set(msg["id"] for msg in CACHED_AGENTS_MESSAGES if msg["id"])

    created_at_str = (datetime.now(timezone.utc)).isoformat()
    new_agent_messages = [
        CachedAgentMessage(
            id=msg.id,
            role=role,
            botId=bot_id,
            runId=run_id,
            createdAt=created_at_str,
            updatedAt=created_at_str,
            messages=msg.model_dump(),
        )
        for msg in messages
        if msg.id and msg.id not in existing_msg_ids
    ]

    CACHED_AGENTS_MESSAGES.extend(new_agent_messages)

    content = json.dumps(CACHED_AGENTS_MESSAGES)

    is_success = await db.redis.set(
        key=f"agent_messages:{run_id}", value=content, ex=60 * 60 * 24
    )

    if not is_success:
        logger.warning(f"Failed to cache agent messages for run_id: {run_id}")
