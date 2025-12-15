from langchain.agents.middleware.types import (
    AgentState,
    _InputAgentState,
    _OutputAgentState,
)
from langgraph.graph.state import CompiledStateGraph
from typing import TypeVar
from src.context import Context

Unknown = TypeVar("Unknown")


LangGraphAgent = CompiledStateGraph[
    AgentState[Unknown], Context, _InputAgentState, _OutputAgentState[Unknown]
]
