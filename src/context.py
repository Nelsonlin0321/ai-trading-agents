import json
from prisma.types import AgentMessageWhereInput
from prisma.models import Run
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from src.db import prisma
from src.typings.context import Context
from src.utils import async_retry

message_type_map = {"ai": AIMessage, "human": HumanMessage, "tool": ToolMessage}


@async_retry(silence_error=False)
async def build_context(run: Run):
    bot = await prisma.bot.find_unique(
        where={"id": run.botId},
        include={
            "user": True,
            "portfolio": {"include": {"positions": True}},
            "watchlist": True,
            "trades": True,
        },
    )
    if not bot:
        raise ValueError(f"Bot with ID {run.botId} not found.")

    if not bot.active:
        raise ValueError(f"Bot with ID {bot.id} is not active.")

    context = Context(run=run, bot=bot, model_name="deepseek")

    return context


@async_retry(silence_error=False)
async def restore_messages(run_id: str) -> list[BaseMessage] | None:
    agent_messages = await prisma.agentmessage.find_many(
        where={"runId": run_id}, order={"createdAt": "asc"}
    )

    if len(agent_messages) == 0:
        return

    latest_cio_msg_idx = None
    for idx, agt_msg in enumerate(agent_messages):
        if agt_msg.role == "CHIEF_INVESTMENT_OFFICER":
            latest_cio_msg_idx = idx

    if latest_cio_msg_idx is None:
        return

    agent_messages = agent_messages[: latest_cio_msg_idx + 1]

    deserialized_messages = [json.loads(agt_msg.messages) for agt_msg in agent_messages]

    #  and make sure the last message is not AI.
    last_msg = deserialized_messages[-1]
    if last_msg["type"] == "ai":
        latest_cio_msg_idx -= 1
        deserialized_messages.pop(-1)

    timestamp = agent_messages[latest_cio_msg_idx].createdAt

    #  delete the unfinished messages after the latest CIO message
    await prisma.agentmessage.delete_many(
        where=AgentMessageWhereInput(runId=run_id, createdAt={"gt": timestamp})
    )

    serialized_messages = [
        message_type_map[msg["type"]](**msg) for msg in deserialized_messages
    ]

    return serialized_messages
