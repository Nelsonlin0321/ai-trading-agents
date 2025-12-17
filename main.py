import os
import sys
import traceback
import asyncio
from loguru import logger
from datetime import datetime, timezone
from prisma.enums import RunStatus
from langchain_core.messages import HumanMessage
from src import secrets  # import secret first before db
from src import db

secrets.load()

ENV = os.environ.get("ENV", "dev")

DEFAULT_USER_PROMPT = """ As AI Agentic Chief Investment Officer,Now,you're tasked to review your portfolio performance,
identify new opportunities, and recommend appropriate actions to optimize portfolio performance aligned with the user's strategy by orchestrating different investment agents.
Now, please review the user's strategy and portfolio performance, and provide your recommendations working with your teams of investment agents.
"""


async def run_agent(run_id: str):
    from src.context import build_context, restore_messages
    from src.agents.chief_investment_officer.agent import (
        build_chief_investment_officer_agent,
    )

    run = await db.prisma.run.find_unique(where={"id": run_id})

    if not run:
        logger.error(f"Run {run_id} not found")
        exit(2)

    logger.info(f"Processing run {run_id} with status {run.status}")

    if run.status == "SUCCESS":
        logger.error(f"Run {run_id} is finished")
        exit(2)

    instruction = run.instruction or DEFAULT_USER_PROMPT
    start_messages = [HumanMessage(content=instruction)]
    deserialized_messages = await restore_messages(run_id)

    if deserialized_messages == "ERROR":
        logger.error(f"Failed to restore messages for run {run_id}")
        exit(2)

    if deserialized_messages:
        start_messages = deserialized_messages

    await db.prisma.run.update(where={"id": run_id}, data={"status": RunStatus.RUNNING})

    context = await build_context(run=run)

    if context == "ERROR":
        logger.error(f"Failed to build context for run {run_id}")
        exit(2)

    agent_graph = await build_chief_investment_officer_agent(context)

    events = agent_graph.astream(
        input={
            "messages": start_messages,  # type: ignore
        },
        context=context,
        stream_mode="values",
    )
    async for event in events:
        if "messages" in event:
            message = event["messages"][-1]
            message.pretty_print()

    await db.prisma.run.update(
        where={"id": run_id},
        data={"status": RunStatus.SUCCESS, "completedAt": datetime.now(timezone.utc)},
    )


async def main(run_id: str):
    try:
        await db.connect()
        await run_agent(run_id)
        logger.info(f"Run {run_id} completed successfully")
    except Exception as e:
        run = await db.prisma.run.find_unique(where={"id": run_id})
        if run:
            await db.prisma.run.update(
                where={"id": run_id}, data={"status": RunStatus.FAILURE}
            )
        logger.error(
            f"Failed to run agent {run_id}: {e}. Traceback: {traceback.format_exc()}"
        )
    finally:
        await db.disconnect()


if __name__ == "__main__":
    logger.info("Starting AI Trading Agents...")
    runId = sys.argv[1]
    asyncio.run(main(runId))
