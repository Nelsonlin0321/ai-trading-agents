from langchain.tools import tool, ToolRuntime
from src.context import Context
from src.tools_adaptors.learning import TakeLearningAct, GetLearningsAct
from src.utils.constants import LEARNING_RATE


take_learning_act = TakeLearningAct()
get_learnings_act = GetLearningsAct()


@tool(take_learning_act.name)
async def take_learning_note(
    note: str,
    runtime: ToolRuntime[Context],
):
    """
    Record a specific, actionable trading insight to improve future decision-making.

    - Noting key insights from research or market observations, rationales, research insights, performance reflections, or strategy
    adjustments to refine judgment, identify patterns in decision-making, reduce biases and continuously improve investment performance.

    This tool is your mechanism for self-evolution. Do not record generic observations (e.g., "Market went down").
    Instead, record specific cause-and-effect lessons that you want your future self to remember.

    Args:
        note: The insightful, actionable lesson to record.
    """
    bot_id = runtime.context.bot.id
    run_id = runtime.context.run.id
    return await take_learning_act.arun(bot_id=bot_id, run_id=run_id, note=note)


@tool(get_learnings_act.name)
async def get_learning_notes(
    runtime: ToolRuntime[Context],
):
    """
    Retrieve past trading insights to inform current decisions and avoid repeating mistakes.

    Before making decisions, consult your "past self" to see what you have learned.

    Use this tool to:
    - Check if a current market setup resembles a past failure or success.
    - Recall specific adjustments you promised to make to your strategy.
    - Validate your current rationale against historical lessons.
    - Ensure you are applying previously learned constraints or opportunities.
    """

    bot_id = runtime.context.bot.id
    return await get_learnings_act.arun(bot_id=bot_id, limit=LEARNING_RATE)
