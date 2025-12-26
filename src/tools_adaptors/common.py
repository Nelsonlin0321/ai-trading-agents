import os
import json
from datetime import datetime
from typing import Literal, TypedDict
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from src import db
from prisma.enums import RunStatus, Role
from prisma.types import AgentMessageWhereInput
from src import utils
from src.models import get_model
from src.utils import async_retry, send_ses_email
from src.tools_adaptors.base import Action


BASE_URL = os.getenv("BASE_URL", "http://localhost:3000")


class GetUserInvestmentStrategyAct(Action):
    @property
    def name(self) -> str:
        return "get_user_investment_strategy"

    @async_retry()
    async def arun(self, bot_id: str) -> str:
        bot = await db.prisma.bot.find_unique(where={"id": bot_id})
        if not bot:
            raise ValueError(f"Bot with ID {bot_id} not found.")

        return bot.strategy


class WriteInvestmentReportEmailAct(Action):
    @property
    def name(self) -> str:
        return "write_investment_report_email"

    @async_retry()
    async def arun(
        self, llm_model: str, botId: str, run_id: str, conversation: str
    ) -> str:
        system_prompt = (
            "You are the Synthesis & Communications Officer (SCO) for an @SandX.AI investment team. "
            "You are a neutral, meticulous scribe and editor whose sole purpose is to accurately capture, structure, "
            "and communicate the team's discussion into a professional investment report. "
            "You have no opinions, no biases, and no agenda beyond faithful representation and clarity. "
        )

        instruction = f"""
        Directly produce a comprehensive, self-contained HTML investment report summarizing the conversation below.

        Requirements:
        1. Start with market analyst's insights to provide a comprehensive updated overview of the market status and trends.
        2. The consolidated investment summary includes insights from every analyst involved in the run,
        presenting each analyst’s investment recommendation together with the detailed rationale
        behind it. It must conclude with the final trading decision (buy, sell, or hold) for
        the tickers based on the trading executor including the rationale that led to that decision.
        
        2.A professionally formatted, self-contained, and stylish HTML string representing the consolidated investment report.
        It must include fully-styled inline CSS (e.g., font-family, color, padding, borders) and pure HTML with CSS to ensure
        consistent rendering across all major email clients. Plain text or Markdown are not acceptable.
        Ensure the content contains:
        - Aggregated insights from all analysts to comprehensively evaluate the tickers.
        - Each analyst’s recommendation and its rationale.
        - Final trading actions (buy/sell/hold) with supporting rationale for each ticker.
        
        3. Must Include A BUTTON with the link  https://sandbox.ai/bots/{botId}/conversation?runId={run_id}
        to allow users to view the full conversation.

        Conversation:
        -------------
        {conversation}
        """

        langchain_model = get_model(llm_model)
        agent = create_agent(
            model=langchain_model,
            system_prompt=system_prompt,
        )
        result = agent.invoke({"messages": [HumanMessage(content=instruction)]})

        report = result["messages"][-1].content

        await db.prisma.run.update(where={"id": run_id}, data={"summary": report})
        return report


class SendInvestmentReportEmailAct(Action):
    @property
    def name(self) -> str:
        return "send_investment_report_email"

    @async_retry()
    async def arun(self, user_id, run_id: str, bot_name: str) -> str:
        run = await db.prisma.run.find_unique(where={"id": run_id})

        if not run:
            return "Run not found"

        investment_report = run.summary
        if not investment_report:
            return "Investment report not found. please write it first."

        date_str = datetime.now().strftime("%Y-%m-%d")

        recipient = await db.prisma.user.find_unique(where={"clerkId": user_id})

        if not recipient:
            return "User not found"

        email = recipient.email
        subject = f"SandX.AI Execution Summary — {bot_name} | {date_str} | {run_id}"

        investment_report = investment_report.strip()
        if investment_report.startswith("```html"):
            investment_report = investment_report["```html".__len__() :]
        if investment_report.startswith("```"):
            investment_report = investment_report[3:]
        if investment_report.endswith("```"):
            investment_report = investment_report[:-3]

        investment_report = investment_report.strip()

        send_ses_email(
            subject=subject,
            recipient=email,
            html_body=investment_report,
        )
        return "Sent investment report email successfully!"


class GetHistoricalReviewedTickersAct(Action):
    @property
    def name(self) -> str:
        return "get_historical_reviewed_tickers"

    @async_retry()
    async def arun(self, botId: str) -> str:
        bot = await db.prisma.bot.find_unique(where={"id": botId})
        if not bot:
            raise ValueError(f"Bot with ID {botId} not found.")

        runs = await db.prisma.run.find_many(
            where={"botId": botId, "status": RunStatus.SUCCESS},
            order={"createdAt": "desc"},
            take=7,
        )
        if not runs:
            return "No previous tickers reviewed found."

        run_tickers = []
        for run in runs:
            if run.tickers:
                tmp_dict = {
                    "tickers": run.tickers,
                    "reviewed_at": run.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
                }
                run_tickers.append(tmp_dict)
        tickers_markdown = utils.dicts_to_markdown_table(run_tickers)
        return tickers_markdown


class GetAnalystAnalysisAct(Action):
    @property
    def name(self) -> str:
        return "get_other_analyst_analysis"

    @async_retry()
    async def arun(
        self,
        role: Literal[
            "MARKET_ANALYST",
            "RISK_ANALYST",
            "EQUITY_RESEARCH_ANALYST",
            "FUNDAMENTAL_ANALYST",
        ],
        run_id: str,
    ) -> str:
        agent_msg_rows = await db.prisma.agentmessage.find_many(
            where=AgentMessageWhereInput(
                role=Role(role),
                runId=run_id,
            ),
            order={"createdAt": "asc"},
        )

        contents = []

        baseMessage = TypedDict(
            "baseMessage",
            {
                "type": Literal["human", "ai", "tool"],
                "content": str,
            },
        )

        for row in agent_msg_rows:
            langchain_message: baseMessage = json.loads(row.messages)

            if langchain_message["type"] == "ai":
                contents.append(langchain_message["content"])

        analysis = "\n".join(contents)
        return analysis
