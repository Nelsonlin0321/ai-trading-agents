import asyncio
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
        await db.connect()
        await asyncio.gather(
            *[
                create_agent_message_if_not_exists(
                    role=self.role,
                    bot_id=context.bot.id,
                    run_id=context.run.id,
                    message=msg,
                )
                for msg in messages
            ]
        )
        await db.disconnect()


@async_retry()
async def create_agent_message_if_not_exists(
    role: AgentRole,
    bot_id: str,
    run_id: str,
    message: AnyMessage,
):
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

    EPHEMERAL_CACHE[message.id] = True


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
