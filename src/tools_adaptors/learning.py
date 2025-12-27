from typing import List, Any, cast
from src import db
from src.utils import async_retry
from src.tools_adaptors.base import Action
from prisma.models import LearningNote
from src.typings import ErrorLiteral


class TakeLearningAct(Action[LearningNote | ErrorLiteral]):
    @property
    def name(self) -> str:
        return "take_learning"

    @async_retry()
    async def arun(self, bot_id: str, run_id: str, note: str) -> LearningNote:  # type: ignore
        """
        Take a learning note for the current run.

        Args:
            bot_id: The ID of the bot.
            run_id: The ID of the current run.
            note: The learning note content.

        Returns:
            The created LearningNote object.
        """
        data = {
            "botId": bot_id,
            "runId": run_id,
            "note": note,
        }
        return await db.prisma.learningnote.create(data=cast(Any, data))


class GetLearningsAct(Action[List[LearningNote] | ErrorLiteral]):
    @property
    def name(self) -> str:
        return "get_learnings"

    @async_retry()
    async def arun(self, bot_id: str, limit: int = 10) -> List[LearningNote]:
        """
        Get past learning notes for the bot.

        Args:
            bot_id: The ID of the bot.
            limit: The maximum number of notes to retrieve. Defaults to 10.

        Returns:
            A list of LearningNote objects.
        """
        return await db.prisma.learningnote.find_many(
            where={
                "botId": bot_id,
            },
            order={
                "createdAt": "desc",
            },
            take=limit,
        )
