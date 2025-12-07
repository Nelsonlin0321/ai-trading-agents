from typing import List, Annotated
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage
from src.context import Context
from src import agents
from src.typings.agent_roles import SubAgentRole
from src.typings.langgraph_agents import LangGraphAgent
from src.utils.message import combine_ai_messages
from src.utils.constants import SPECIALIST_ROLES


agent_building_map = {
    "MARKET_ANALYST": agents.build_market_analyst_agent,
    "RISK_ANALYST": agents.build_risk_analyst_agent,
    "EQUITY_RESEARCH_ANALYST": agents.build_equity_research_analyst_agent,
    "FUNDAMENTAL_ANALYST": agents.build_fundamental_analyst_agent,
    "TRADING_EXECUTOR": agents.build_trading_executor_agent,
}


@tool("handoff_to_specialist")
async def handoff_to_specialist(
    role: Annotated[
        SubAgentRole,
        f"Role of the specialist agent to handoff to. Must be one of: {' | '.join(SPECIALIST_ROLES)}",
    ],
    task: Annotated[
        str,
        "A clear, detailed description of the task the specialist should perform. Include relevant context, and expected deliverables to ensure the specialist can act effectively.",
    ],
    runtime: ToolRuntime[Context],
) -> str:
    """
    Delegate a specific investment-related task to a specialist agent and return the agent's response.

    This function:
    1. Validates that the requested role is one of the registered specialist roles.
    2. Combines the agent's response messages into a single string and returns it.

    The returned string consolidates all AI-generated messages from the specialist, providing
    a concise summary or detailed output depending on the task performed.

    Args:
        role: Role of the specialist agent to handoff to.
        task: A clear, detailed description of the task the specialist should perform. Include relevant context, and expected deliverables to ensure the specialist can act effectively.

    Returns:
        A single string containing the combined AI messages from the specialist agent's response.

    Raises:
        ValueError: If the provided role is not in the list of valid specialist roles.
    """
    context = runtime.context
    if role not in agent_building_map:
        raise ValueError(f"Invalid role: {role}. Must be one of: {SPECIALIST_ROLES}")

    agent: LangGraphAgent = await agent_building_map[role](context)

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Task from the chief investment officer to you: {task}",
                }
            ]
        },
        context=context,
    )

    messages: List[BaseMessage] = response["messages"]

    content = combine_ai_messages(messages)

    return content


__all__ = ["handoff_to_specialist"]
