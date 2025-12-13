from src import db
from datetime import datetime
from src.utils import async_retry, send_ses_email
from src.tools_adaptors.base import Action


class GetUserInvestmentStrategyAct(Action):
    @property
    def name(self) -> str:
        return "get_user_investment_strategy"

    @async_retry()
    async def arun(self, botId: str) -> str:
        bot = await db.prisma.bot.find_unique(where={"id": botId})
        if not bot:
            raise ValueError(f"Bot with ID {botId} not found.")

        return bot.strategy


class SendInvestmentReportEmailAct(Action):
    @property
    def name(self) -> str:
        return "write_and_send_investment_report_email"

    @async_retry()
    async def arun(
        self, user_id: str, bot_name: str, run_id: str, investment_summary: str
    ) -> str:
        await db.prisma.run.update(
            where={"id": run_id}, data={"summary": investment_summary}
        )

        date_str = datetime.now().strftime("%Y-%m-%d")

        recipient = await db.prisma.user.find_unique(where={"id": user_id})

        if not recipient:
            return "User not found"

        email = recipient.email
        subject = f"SandX.AI Execution Summary — {bot_name} | {date_str} | {run_id}"
        send_ses_email(
            subject=subject,
            recipient=email,
            html_body=investment_summary,
        )
        return "Successfully written and sent investment report email!"
