"""Define structured observability events for autonomous agent execution."""

import json
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """Provide fields shared by every runtime observability event."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data_source: str
    invocation_count: int = Field(ge=0)
    tool_count: int = Field(ge=0)

    @property
    def event_name(self) -> str:
        """Return the concrete event discriminator used by console reporters."""
        return str(self.model_dump()["event"])


class AgentActionSelection(BaseModel):
    """Describe one typed action selected within an agent reasoning turn."""

    tool: str
    action_model: str
    arguments: dict[str, Any]


class AgentDecisionEvent(AgentEvent):
    """Describe the validated action batch selected by the agent."""

    event: Literal["agent_decision"] = "agent_decision"
    rationale: str
    actions: list[AgentActionSelection] = Field(min_length=1)
    parallel: bool
    available_tools: list[str]
    source_state: dict[str, Any]


class ToolResultEvent(AgentEvent):
    """Describe one executed or rejected tool and the resulting durable state."""

    event: Literal["tool_result"] = "tool_result"
    tool: str
    action_model: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    outcome: str
    source_state: dict[str, Any]
    requested_wait_seconds: int | None = Field(default=None, ge=1)
    elapsed_wait_seconds: float | None = Field(default=None, ge=0)


class ApprovalRequestedEvent(AgentEvent):
    """Describe a source action suspended for an operator decision."""

    event: Literal["approval_requested"] = "approval_requested"
    approval_id: str
    title: str
    body: str
    resource_ids: list[str]
    tool: str
    action_model: str
    arguments: dict[str, Any]
    delivery: dict[str, Any]


class ApprovalResolvedEvent(AgentEvent):
    """Record the operator decision that resumed a suspended source action."""

    event: Literal["approval_resolved"] = "approval_resolved"
    approval_id: str
    decision: Literal["approved", "rejected"]
    tool: str
    action_model: str
    arguments: dict[str, Any]


class InvocationFinishedEvent(AgentEvent):
    """Describe why one graph invocation returned control to the runtime."""

    event: Literal["invocation_finished"] = "invocation_finished"
    summary: str
    outcome: str
    source_state: dict[str, Any]


class AgentReporter(Protocol):
    """Consume structured runtime events without affecting agent decisions."""

    def emit(self, event: AgentEvent) -> None:
        """Publish one immutable snapshot of agent execution state."""
        ...


class CompositeAgentReporter:
    """Fan out each typed runtime event to multiple reporting destinations."""

    def __init__(self, *reporters: AgentReporter) -> None:
        """Store reporters in deterministic delivery order."""
        self._reporters = reporters

    def emit(self, event: AgentEvent) -> None:
        """Deliver one event to every configured reporter."""
        for reporter in self._reporters:
            reporter.emit(event)


class ConsoleAgentReporter:
    """Write compact structured events while retaining full data for other reporters."""

    def emit(self, event: AgentEvent) -> None:
        """Render an event for humans, log collectors, and Docker inspection."""
        payload = event.model_dump(mode="json")
        source_state = payload.get("source_state")
        if isinstance(source_state, dict):
            payload["source_state"] = _source_state_summary(source_state)
        result = payload.get("result")
        if isinstance(result, dict):
            payload["result"] = _result_summary(result)
        print(f"[agent.{event.event_name}] {json.dumps(payload, separators=(',', ':'))}")


def _source_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Summarize common source state without assuming a particular platform."""
    summary: dict[str, Any] = {
        key: state[key]
        for key in ("phase", "objective", "objective_waited", "poll_interval_seconds")
        if key in state
    }
    for key, value in state.items():
        if key in summary:
            continue
        if isinstance(value, dict):
            summary[f"{key}_keys"] = list(value)
        elif isinstance(value, list) or value is not None:
            summary[key] = value
    return summary


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Collapse collection payloads while retaining counts, identifiers, and scalar evidence."""
    summary: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
            identifiers = [
                item.get("id") for item in value if isinstance(item, dict) and item.get("id")
            ]
            if identifiers:
                summary[f"{key}_ids"] = identifiers
        elif isinstance(value, dict):
            summary[f"{key}_keys"] = list(value)
        else:
            summary[key] = value
    return summary
