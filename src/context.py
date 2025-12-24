import json
from prisma.types import AgentMessageWhereInput
from prisma.models import Run
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.db import prisma, CachedAgentMessage, CACHED_AGENTS_MESSAGES
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


# @async_retry(silence_error=False)
async def restore_messages(
    run_id: str,
) -> list[HumanMessage | AIMessage | ToolMessage] | None:
    agent_message_rows = await prisma.agentmessage.find_many(
        where={"runId": run_id}, order={"createdAt": "asc"}
    )

    if len(agent_message_rows) == 0:
        return

    #  only Retry CIO messages
    cio_agent_msg_rows = [
        msg_row
        for msg_row in agent_message_rows
        if msg_row.role == "CHIEF_INVESTMENT_OFFICER"
    ]

    deserialized_messages = [
        json.loads(msg_row.messages) for msg_row in cio_agent_msg_rows
    ]

    # Trim trailing AI messages iteratively
    while deserialized_messages and deserialized_messages[-1]["type"] == "ai":
        deserialized_messages.pop(-1)
        cio_agent_msg_rows.pop(-1)

    if len(deserialized_messages) == 0:
        await prisma.agentmessage.delete_many(
            where=AgentMessageWhereInput(runId=run_id)
        )
        return

    timestamp = cio_agent_msg_rows[-1].createdAt
    #  delete the unfinished messages after the latest CIO message
    await prisma.agentmessage.delete_many(
        where=AgentMessageWhereInput(
            runId=run_id, createdAt={"gt": timestamp}
        )  # TO CHECK
    )

    updated_cached_messages: list[CachedAgentMessage] = []
    for row in cio_agent_msg_rows:
        updated_cached_messages.append(
            CachedAgentMessage(
                id=row.id,
                role=row.role,  # pyright: ignore
                botId=row.botId,
                runId=run_id,
                createdAt=(row.createdAt).isoformat(),
                updatedAt=(row.updatedAt).isoformat(),
                messages=json.loads(row.messages),
            )
        )

    CACHED_AGENTS_MESSAGES.extend(updated_cached_messages)
    _ = json.dumps(CACHED_AGENTS_MESSAGES)  # make sure it's able to serialize

    serialized_messages: list[HumanMessage | AIMessage | ToolMessage] = [
        message_type_map[msg["type"]](**msg) for msg in deserialized_messages
    ]

    return serialized_messages


if __name__ == "__main__":
    #  python -m src.context
    import asyncio
    from src import db

    run_id = "ee19573e-60db-46c8-a016-f1c4b41d84ff"

    async def test_restore_messages():
        await db.connect()
        await restore_messages(run_id)
        await db.disconnect()

    asyncio.run(test_restore_messages())
