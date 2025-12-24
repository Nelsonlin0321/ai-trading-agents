import json
from prisma.types import AgentMessageWhereInput
from prisma.models import Run
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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

    context = Context(run=run, bot=bot, llm_model=bot.llmModel or "deepseek-v3.2")

    return context


@async_retry(silence_error=False)
async def restore_messages(
    run_id: str,
) -> list[HumanMessage | AIMessage | ToolMessage] | None:
    agent_messages = await prisma.agentmessage.find_many(
        where={"runId": run_id}, order={"createdAt": "asc"}
    )

    if len(agent_messages) == 0:
        return

    #  only Retry CIO messages
    cio_agent_msgs = [
        agt_msg
        for agt_msg in agent_messages
        if agt_msg.role == "CHIEF_INVESTMENT_OFFICER"
    ]

    deserialized_messages = [json.loads(agt_msg.messages) for agt_msg in cio_agent_msgs]

    last_msg = deserialized_messages[-1]
    while last_msg["type"] == "ai":
        deserialized_messages.pop(-1)
        last_msg = deserialized_messages[-1]

    last_msg = deserialized_messages[-1]

    timestamp = cio_agent_msgs[-1].createdAt

    #  delete the unfinished messages after the latest CIO message
    await prisma.agentmessage.delete_many(
        where=AgentMessageWhereInput(runId=run_id, createdAt={"gt": timestamp})
    )

    serialized_messages: list[HumanMessage | AIMessage | ToolMessage] = [
        message_type_map[msg["type"]](**msg) for msg in deserialized_messages
    ]

    return serialized_messages
