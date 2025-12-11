import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Any
from langchain_core.messages import AnyMessage
from langgraph.runtime import Runtime
from langchain.agents import middleware
from prisma.types import (
    AgentMessageCreateInput,
)
from src import db
from src.typings.context import Context
from src.models import get_model
from src.utils.constants import ALL_ROLES
from src.typings.agent_roles import AgentRole
from src.utils import async_retry

langchain_model = get_model("deepseek")

EPHEMERAL_CACHE = {}

summarization_middleware = middleware.SummarizationMiddleware(
    model=langchain_model,
    max_tokens_before_summary=128_000,
    messages_to_keep=20,
)


todo_list_middleware = middleware.TodoListMiddleware()


class LoggingMiddleware(middleware.AgentMiddleware[middleware.AgentState, Context]):
    def __init__(self, role: AgentRole):
        assert role in ALL_ROLES
        self.role: AgentRole = role

    async def aafter_model(
        self, state: middleware.AgentState, runtime: Runtime[Context]
    ) -> None:
        context = runtime.context
        messages = state["messages"]
        # Fire-and-forget: schedule persistence in the background without awaiting

        asyncio.create_task(
            persist_agent_messages(  # type: ignore
                role=self.role,
                bot_id=context.bot.id,
                run_id=context.run.id,
                messages=messages,
            )
        )

        # await cache_agent_messages(
        #     role=self.role,
        #     bot_id=context.bot.id,
        #     run_id=context.run.id,
        #     messages=messages,
        # )


@async_retry()
async def persist_agent_message(
    role: AgentRole, bot_id: str, run_id: str, message: AnyMessage
):
    if not message.id:
        return

    if message.id in EPHEMERAL_CACHE:
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


@async_retry()
async def persist_agent_messages(
    role: AgentRole, bot_id: str, run_id: str, messages: list[AnyMessage]
):
    await asyncio.gather(
        *[
            persist_agent_message(
                role=role,
                bot_id=bot_id,
                run_id=run_id,
                message=msg,
            )
            for msg in messages
        ]
    )


CachedAgentMessage = TypedDict(
    "CachedAgentMessage",
    {
        "id": str,
        "role": AgentRole,
        "botId": str,
        "runId": str,
        "createdAt": str,
        "updatedAt": str,
        "messages": dict[str, Any],
    },
)


@async_retry()
async def cache_agent_messages(
    role: AgentRole, bot_id: str, run_id: str, messages: list[AnyMessage]
):
    delta_seconds = 10
    now = datetime.now(timezone.utc) - timedelta(seconds=delta_seconds * len(messages))

    agent_messages = [
        CachedAgentMessage(
            id=msg.id,
            role=role,
            botId=bot_id,
            runId=run_id,
            createdAt=(now + timedelta(seconds=delta_seconds * idx)).isoformat(),
            updatedAt=(now + timedelta(seconds=delta_seconds * idx)).isoformat(),
            messages=msg.model_dump(),
        )
        for (idx, msg) in enumerate(messages)
        if msg.id
    ]

    content = json.dumps(agent_messages)

    is_success = await db.redis.set(
        key=f"agent_messages:{run_id}", value=content, ex=60 * 60 * 24
    )
    assert is_success


# await db.redis()


# class ExampleLoggingMiddleware(middleware.AgentMiddleware):

#     async def aafter_model(
#         self, state: middleware.AgentState, runtime: Runtime
#     ) -> None:
#         messages = state["messages"]
#         agent_message_id = messages[0].id
#         print(f"agent_message_id: {agent_message_id}")

#         __all__ = [
#             "summarization_middleware",
#             "todo_list_middleware",
#             "LoggingMiddleware",
#             "ExampleLoggingMiddleware"
#         ]
