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


async def send_summary_email(runId: str):
    response = await SandxAPIClient[EmailResult].get(
        f"/run/send-summary-report?runId={runId}"
    )
    return response
