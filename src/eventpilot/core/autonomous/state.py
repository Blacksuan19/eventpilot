"""Shared state and callable contracts for the autonomous agent graph."""

from collections.abc import Awaitable, Callable
from operator import add
from typing import Annotated, Any, Literal, TypedDict


class AutonomousAgentState(TypedDict, total=False):
    """Persist generic loop state and opaque data-source state on one thread."""

    transcript: list[dict[str, Any]]
    turn: dict[str, Any]
    source_state: dict[str, Any]
    outcome: str
    invocation_summary: str | None
    invocation_count: int
    tool_count: int
    pending_approval: dict[str, Any] | None
    approval_decision: str | None
    pending_wait: dict[str, Any] | None
    parallel_action_index: int
    parallel_results: Annotated[list[dict[str, Any]], add]


Sleep = Callable[[float], Awaitable[None]]
ToolRoute = Literal[
    "source_tool",
    "retryable_source_tool",
    "select_objective",
    "request_approval",
    "send_alert",
    "wait",
    "reject_action",
]
