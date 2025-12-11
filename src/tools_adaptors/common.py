from src import db
from src.utils import async_retry
from src.tools_adaptors.base import Action
from src.services.sandx_ai import send_summary_email


class GetUserInvestmentStrategyAct(Action):
    @property
    def name(self) -> str:
        return "get_user_investment_strategy"

    @async_retry()
    async def arun(self, botId: str) -> str:
        await db.connect()
        bot = await db.prisma.bot.find_unique(where={"id": botId})
        if not bot:
            raise ValueError(f"Bot with ID {botId} not found.")
        await db.disconnect()
        return bot.strategy


class SendInvestmentReportEmailAct(Action):
    @property
    def name(self) -> str:
        return "send_investment_report_email"

    @async_retry()
    async def arun(self, run_id: str, investment_summary: str) -> str:
        await db.connect()
        await db.prisma.run.update(
            where={"id": run_id}, data={"summary": investment_summary}
        )
        response = await send_summary_email(run_id)
        if not response["success"]:
            return f"Failed to send email: {response['message']}"

        return response.get("message", "Email sent successfully")
