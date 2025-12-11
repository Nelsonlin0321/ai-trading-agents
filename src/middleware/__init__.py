import asyncio
import json
import pickle
from loguru import logger
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

PERSISTED_MSG_IDS: set[str] = set()
MESSAGES_STATE_PATH = "all_agents_messages_state.pkl"

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


def load_messages_state() -> list[AnyMessage]:
    try:
        with open(MESSAGES_STATE_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []


def save_messages_state(cached_messages: list[AnyMessage]) -> None:
    with open(MESSAGES_STATE_PATH, "wb") as f:
        pickle.dump(cached_messages, f)


langchain_model = get_model("deepseek")


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

    #  Only one agent called this middleware at a time
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
    existing_messages = load_messages_state()
    existing_msg_ids = set(msg.id for msg in existing_messages if msg.id)
    delta_messages = [
        msg for msg in messages if msg.id and msg.id not in existing_msg_ids
    ]

    existing_messages.extend(delta_messages)
    complete_messages = existing_messages.copy()
    save_messages_state(complete_messages)

    delta_seconds = 10
    now = datetime.now(timezone.utc) - timedelta(
        seconds=delta_seconds * len(complete_messages)
    )
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
        for (idx, msg) in enumerate(complete_messages)
        if msg.id
    ]

    content = json.dumps(agent_messages)

    is_success = await db.redis.set(
        key=f"agent_messages:{run_id}", value=content, ex=60 * 60 * 24
    )

    if not is_success:
        logger.warning(f"Failed to cache agent messages for run_id: {run_id}")
