from typing import List, Annotated, Any
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
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

tool_name_template = "handoff_to_{role_name}"
description_template = "Delegate a specific investment-related task to the {role_name} to get the {role_name}'s analysis result."
task_parameter_annotation = (
    "A clear, detailed description of the task the specialist should perform."
    "Include relevant context, and expected deliverables to ensure the specialist can act effectively.",
    "In addition to their analysis, Ask them their recommendation and state BUY, SELL, or HOLD with your rationale, Allocation Percentage, and confidence level (0.0-1.0)",
)


@tool(
    tool_name_template.format(role_name="market_analyst"),
    description=description_template.format(role_name="market_analyst"),
)
async def handoff_to_market_analyst(
    task: Annotated[
        str,
        task_parameter_annotation,
    ],
    runtime: ToolRuntime[Context],
) -> str:
    content = await handoff_to_specialist_func("MARKET_ANALYST", task, runtime.context)
    return content


@tool(
    tool_name_template.format(role_name="equity_selection_analyst"),
    description=description_template.format(role_name="equity_selection_analyst"),
)
async def handoff_to_equity_selection_analyst(
    task: Annotated[
        str,
        task_parameter_annotation,
    ],
    runtime: ToolRuntime[Context],
) -> str:
    content = await handoff_to_specialist_func(
        "EQUITY_SELECTION_ANALYST", task, runtime.context
    )
    return content


@tool(
    tool_name_template.format(role_name="risk_analyst"),
    description=description_template.format(role_name="risk_analyst"),
)
async def handoff_to_risk_analyst(
    task: Annotated[
        str,
        task_parameter_annotation,
    ],
    runtime: ToolRuntime[Context],
) -> str:
    content = await handoff_to_specialist_func("RISK_ANALYST", task, runtime.context)
    return content


@tool(
    tool_name_template.format(role_name="fundamental_analyst"),
    description=description_template.format(role_name="fundamental_analyst"),
)
async def handoff_to_fundamental_analyst(
    task: Annotated[
        str,
        task_parameter_annotation,
    ],
    runtime: ToolRuntime[Context],
) -> str:
    content = await handoff_to_specialist_func(
        "FUNDAMENTAL_ANALYST", task, runtime.context
    )
    return content


@tool(
    tool_name_template.format(role_name="technical_analyst"),
    description=description_template.format(role_name="technical_analyst"),
)
async def handoff_to_technical_analyst(
    task: Annotated[
        str,
        task_parameter_annotation,
    ],
    runtime: ToolRuntime[Context],
) -> str:
    content = await handoff_to_specialist_func(
        "TECHNICAL_ANALYST", task, runtime.context
    )
    return content


@tool(
    tool_name_template.format(role_name="equity_research_analyst"),
    description=description_template.format(role_name="equity_research_analyst"),
)
async def handoff_to_equity_research_analyst(
    task: Annotated[
        str,
        task_parameter_annotation,
    ],
    runtime: ToolRuntime[Context],
) -> str:
    content = await handoff_to_specialist_func(
        "EQUITY_RESEARCH_ANALYST", task, runtime.context
    )
    return content


@tool(
    tool_name_template.format(role_name="trading_executor"),
    description=description_template.format(role_name="trading_executor"),
)
async def handoff_to_trading_executor(
    task: Annotated[
        str,
        task_parameter_annotation,
    ],
    runtime: ToolRuntime[Context],
) -> str:
    content = await handoff_to_specialist_func(
        "TRADING_EXECUTOR", task, runtime.context
    )
    return content


def load_historical_conversation(
    role: SubAgentRole,
) -> list[HumanMessage | AIMessage | ToolMessage]:
    historical_messages = [
        msg["messages"] for msg in CACHED_AGENTS_MESSAGES if msg["role"] == role
    ]
    deserialized_messages: list[HumanMessage | AIMessage | ToolMessage] = [
        message_type_map[msg["type"]](**msg) for msg in historical_messages
    ]
    return deserialized_messages


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
        input_messages.extend(load_historical_conversation(role))
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


# def get_handoff_agent_tools():
#     tools: List[BaseTool] = []
#     for role, _ in AGENT_BUILDING_MAP.items():
#         role: SubAgentRole = role
#         role_name = role.lower()
#         role_name = "_".join([word.capitalize() for word in role_name.split("_")])

#         description = f"""
#         Delegate, handoff a specific investment-related task to the {role_name} to get the {role_name}'s analysis result.
#         """

#         @tool(f"handoff_to_{role_name}", description=description)
#         async def handoff_to_specialist(
#             task: Annotated[
#                 str,
#                 "A clear, detailed description of the task the specialist should perform. Include relevant context, and expected deliverables to ensure the specialist can act effectively.",
#             ],
#             runtime: ToolRuntime[Context],
#         ) -> str:
#             content = await handoff_to_specialist_func(role, task, runtime.context)
#             return content

#         tools.append(handoff_to_specialist)

#     return tools

__all__ = [
    "handoff_to_market_analyst",
    "handoff_to_equity_selection_analyst",
    "handoff_to_risk_analyst",
    "handoff_to_fundamental_analyst",
    "handoff_to_technical_analyst",
    "handoff_to_equity_research_analyst",
    "handoff_to_trading_executor",
]
