from typing import List, Annotated, Any
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools.base import BaseTool
from src.context import Context
from src import agents
from src.context import message_type_map
from src.db import CACHED_AGENTS_MESSAGES
from src.typings.agent_roles import SubAgentRole
from src.typings.langgraph_agents import LangGraphAgent
from src.utils.message import combine_ai_messages


AGENT_BUILDING_MAP: dict[SubAgentRole, Any] = {
    "MARKET_ANALYST": agents.build_market_analyst_agent,
    "RISK_ANALYST": agents.build_risk_analyst_agent,
    "EQUITY_RESEARCH_ANALYST": agents.build_equity_research_analyst_agent,
    "FUNDAMENTAL_ANALYST": agents.build_fundamental_analyst_agent,
    "TRADING_EXECUTOR": agents.build_trading_executor_agent,
    "TECHNICAL_ANALYST": agents.build_technical_analyst_agent,
    "EQUITY_SELECTION_ANALYST": agents.build_equity_selection_analyst_agent,
}

REQUIRED_HISTORICAL_CONVERSATION_AGENTS = {"EQUITY_SELECTION_ANALYST"}


def get_handoff_agent_tools():
    tools: List[BaseTool] = []
    for role, _ in AGENT_BUILDING_MAP.items():
        role: SubAgentRole = role
        role_name = role.lower()
        role_name = "_".join([word.capitalize() for word in role_name.split("_")])

        description = f"""
        Delegate, handoff a specific investment-related task to the {role_name} to get the {role_name}'s analysis result.
        """

        @tool(f"handoff_to_{role_name}", description=description)
        async def handoff_to_specialist(
            task: Annotated[
                str,
                "A clear, detailed description of the task the specialist should perform. Include relevant context, and expected deliverables to ensure the specialist can act effectively.",
            ],
            runtime: ToolRuntime[Context],
        ) -> str:
            content = await handoff_to_specialist_func(role, task, runtime.context)
            return content

        tools.append(handoff_to_specialist)

    return tools


async def handoff_to_specialist_func(
    role: SubAgentRole,
    task: str,
    context: Context,
) -> str:
    agent: LangGraphAgent = await AGENT_BUILDING_MAP[role](context)

    default_message = HumanMessage(
        content=f"Task from the chief investment officer to you: {task}"
    )

    input_messages: list[HumanMessage | AIMessage | ToolMessage] = []
    if role in REQUIRED_HISTORICAL_CONVERSATION_AGENTS:
        historical_messages = [
            msg["messages"] for msg in CACHED_AGENTS_MESSAGES if msg["role"] == role
        ]
        deserialized_messages: list[HumanMessage | AIMessage | ToolMessage] = [
            message_type_map[msg["type"]](**msg) for msg in historical_messages
        ]
        input_messages.extend(deserialized_messages)
    input_messages.append(default_message)

    response = await agent.ainvoke(
        {
            # pyright: ignore [reportArgumentType]
            "messages": input_messages
        },
        context=context,
    )

    messages: List[BaseMessage] = response["messages"]

    content = combine_ai_messages(messages)

    return content


__all__ = ["get_handoff_agent_tools"]
