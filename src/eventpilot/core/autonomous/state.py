"""Shared state and callable contracts for the autonomous agent graph."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict


class AutonomousAgentState(TypedDict, total=False):
    """Persist generic loop state and opaque data-source state on one thread."""

    transcript: list[dict[str, Any]]
    turn: dict[str, Any]
    source_state: dict[str, Any]
    outcome: str
    cycle_summary: str | None
    cycle_count: int
    tool_count: int
    pending_approval: dict[str, Any] | None
    approval_decision: str | None


Sleep = Callable[[float], Awaitable[None]]
ToolRoute = Literal[
    "source_tool",
    "select_objective",
    "request_approval",
    "send_alert",
    "wait",
    "finish_cycle",
    "reject_action",
]
