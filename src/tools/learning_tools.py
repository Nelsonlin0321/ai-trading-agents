from langchain.tools import tool, ToolRuntime
from src.context import Context
from src.tools_adaptors.learning import TakeLearningAct, GetLearningsAct

take_learning_act = TakeLearningAct()
get_learnings_act = GetLearningsAct()


@tool(take_learning_act.name)
async def take_learning_note(
    note: str,
    runtime: ToolRuntime[Context],
):
    """
    Take a learning note for the current run to learn from the past.

    Use this tool to:
    - Record key insights, lessons learned, or observations from the current trading session.
    - Store knowledge that can be useful for future decision-making.
    - Document reasons for success or failure of specific strategies or actions.

    Args:
        note: The content of the learning note.
    """
    bot_id = runtime.context.bot.id
    run_id = runtime.context.run.id
    return await take_learning_act.arun(bot_id=bot_id, run_id=run_id, note=note)


@tool(get_learnings_act.name)
async def get_learning_notes(
    runtime: ToolRuntime[Context],
    limit: int = 10,
):
    """
    Get past learning notes for the bot to improve future performance.

    Use this tool to:
    - Review past lessons learned and insights.
    - Avoid repeating past mistakes.
    - Adapt strategies based on historical observations.

    Args:
        limit: The maximum number of notes to retrieve. Defaults to 10.
    """
    bot_id = runtime.context.bot.id
    return await get_learnings_act.arun(bot_id=bot_id, limit=limit)
