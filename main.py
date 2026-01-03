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

DEFAULT_USER_PROMPT = """As the Chief Investment Officer, you are tasked with a comprehensive review and optimization of the portfolio.
Your objective is to execute a disciplined investment by following this step-by-step framework for every run. Do not skip steps or change the order.

STEP 0: PORTFOLIO & STRATEGY REVIEW
- Review the current portfolio performance and confirm alignment with the user's investment strategy.
STEP 1: MARKET ANALYSIS
- Delegate the initial market analysis to the [Market Analyst] to provide you with a broad market overview and identify key trends. Wait for their report before proceeding.
STEP 2: EQUITIES (TICKERS) SELECTION
- Call 'get_selected_tickers' to get the list of selected tickers if the user didn't specify any tickers, otherwise you will continue with the tickers specified by the user without delegating to the equity selection analyst.
If the list is empty or the user didn't specify any tickers, delegate ticker selection to the equity selection analyst.
Before delegating to the equity selection analyst, please ensure market analyst has provided a market analysis to you.
STEP 3: DEEP DIVE ANALYSIS (Per Ticker)
For each selected ticker, execute the following delegation in parallel:
  - 3.1 [Equity Research Analyst if available]: Request current news and narrative analysis with BUY/SELL/HOLD recommendation.
  - 3.2 [Fundamental Analyst if available]: Request valuation and financial health analysis with BUY/SELL/HOLD recommendation.
  - 3.3 [Technical Analyst if available]: Request technical analysis with BUY/SELL/HOLD recommendation.
  - 3.4 [Risk Analyst if available]: Request risk assessment and position limit checks with BUY/SELL/HOLD recommendation.
  - 3.5 SYNTHESIS: Combine these 4 analyses' results into a final BUY/SELL/HOLD recommendation with a specific rationale and confidence score aligning.
STEP 4: TRADE EXECUTION
- If the market is open and you have high-confidence recommendations (BUY/SELL), delegate execution to the [Trading Executor].
- Provide clear and detailed instructions summary including all tickers your recommended (Ticker, Action, Quantity/Allocation, Confidence Score, detailed Rationale).
STEP 5: FINAL REPORTING
- Compile all findings, rationales, and execution results.
- Send an investment recommendation summary email to the user.
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

    logger.info(f"Instruction: {run.instruction}")
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
        exit(0)
    except Exception as e:
        run = await db.prisma.run.find_unique(where={"id": run_id})
        if run:
            await db.prisma.run.update(
                where={"id": run_id}, data={"status": RunStatus.FAILURE}
            )
        logger.error(
            f"Failed to run agent {run_id}: {e}. Traceback: {traceback.format_exc()}"
        )
        exit(1)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    logger.info("Starting AI Trading Agents...")
    runId = sys.argv[1]
    asyncio.run(main(runId))
