from src import db
from src.utils import async_retry, dicts_to_markdown_table
from src.tools_adaptors.base import Action
from prisma.types import LearningNoteCreateInput
from src.typings import ErrorLiteral


class TakeLearningAct(Action[str | ErrorLiteral]):
    @property
    def name(self) -> str:
        return "take_learning_notes"

    @async_retry()
    async def arun(self, bot_id: str, run_id: str, note: str):
        await db.prisma.learningnote.create(
            data=LearningNoteCreateInput(botId=bot_id, runId=run_id, note=note)
        )
        return "Taking learning note successfully: " + note


class GetLearningsAct(Action[str | ErrorLiteral]):
    @property
    def name(self) -> str:
        return "get_learning_notes"

    @async_retry()
    async def arun(self, bot_id: str, limit: int = 10) -> str:
        learning_notes = await db.prisma.learningnote.find_many(
            where={
                "botId": bot_id,
            },
            order={
                "createdAt": "desc",
            },
            take=limit,
        )

        notes = [
            {
                "create_date": note.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
                "note": note.note,
            }
            for note in learning_notes
        ]

        notes_markdown = dicts_to_markdown_table(notes)
        return notes_markdown
