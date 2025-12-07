import json
from langgraph.runtime import Runtime
from langchain.agents import middleware
from prisma.types import (
    AgentMessageUpsertInput,
    AgentMessageCreateInput,
    AgentMessageUpdateInput,
)
from src import db
from src.typings.context import Context
from src.models import get_model
from src.utils.constants import ALL_ROLES
from src.typings.agent_roles import AgentRole

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
        self.role = role

    async def aafter_model(
        self, state: middleware.AgentState, runtime: Runtime[Context]
    ) -> None:
        context = runtime.context
        messages = state["messages"]
        agent_message_id = messages[0].id
        message_raw_list = [msg.model_dump_json() for msg in messages]
        messages_json = json.dumps(message_raw_list)

        await db.connect()

        await db.prisma.agentmessage.upsert(
            where={"id": agent_message_id},
            data=AgentMessageUpsertInput(
                create=AgentMessageCreateInput(
                    role=self.role,
                    botId=context.bot.id,
                    messages=messages_json,
                    runId=context.run.id,
                ),
                update=AgentMessageUpdateInput(messages=messages_json),
            ),
        )

        await db.disconnect()


__all__ = [
    "summarization_middleware",
    "todo_list_middleware",
    "LoggingMiddleware",
]
