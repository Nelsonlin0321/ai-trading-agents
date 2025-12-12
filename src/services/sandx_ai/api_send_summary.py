from typing import TypedDict
from src.services.sandx_ai.api_client import SandxAPIClient


EmailResult = TypedDict(
    "EmailResult",
    {
        "success": bool,
        "status": int,
        "message": str,
        "messageId": str | None,
    },
)

api_client = SandxAPIClient[EmailResult]("/run/send-summary-report")


async def send_summary_email(runId: str):
    response = await api_client.get(params={"runId": runId})
    return response
