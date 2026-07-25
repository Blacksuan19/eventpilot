"""Define the contract implemented by autonomous monitoring data sources."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from eventpilot.core.approvals import ApprovalRequest


class SourceToolCall(BaseModel):
    """Provide the common discriminator required by every source tool."""

    @property
    def tool_name(self) -> str:
        """Return the validated discriminator declared by a concrete tool model."""
        return str(self.model_dump()["tool"])


@dataclass(frozen=True, slots=True)
class SourceContext:
    """Expose durable source state and safe runtime services to a tool handler."""

    state: dict[str, Any]
    transcript: list[dict[str, Any]]
    clock: Callable[[], float]
    max_tool_calls_per_cycle: int


@dataclass(frozen=True, slots=True)
class SourceExecution:
    """Return one JSON tool result and the source's next durable state."""

    result: dict[str, Any]
    state: dict[str, Any]


class DataSource(Protocol):
    """Supply typed tools and deterministic policy for one monitored platform."""

    name: str
    instructions: str
    tool_types: tuple[type[SourceToolCall], ...]

    def initial_state(self) -> dict[str, Any]:
        """Return the source-owned state used for a new supervisor thread."""
        ...

    def available_tools(self, state: dict[str, Any]) -> set[str]:
        """Return source tools allowed by the current deterministic state."""
        ...

    def is_idle(self, state: dict[str, Any]) -> bool:
        """Return whether no source work is currently running or actionable."""
        ...

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate a persisted tool payload against the source's tool catalog."""
        ...

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Execute one source tool and return its result and durable state update."""
        ...

    def after_wait(
        self, state: dict[str, Any], *, requested_seconds: int, wake_at: float
    ) -> dict[str, Any]:
        """Update source scheduling state after the generic wait tool completes."""
        ...

    def validate_wait(self, state: dict[str, Any]) -> str | None:
        """Return a rejection reason when source state requires immediate action."""
        ...

    def validate_alert(self, resource_ids: list[str], state: dict[str, Any]) -> str | None:
        """Return a rejection reason when an alert lacks source evidence or scope."""
        ...

    def record_alert(
        self, resource_ids: list[str], state: dict[str, Any], *, delivered_at: float
    ) -> dict[str, Any]:
        """Persist source-owned delivery and monitoring state after an alert succeeds."""
        ...

    def should_continue_after_alert(self, state: dict[str, Any]) -> bool:
        """Return whether delivered work leaves useful work in the current objective."""
        ...

    def validate_finish(
        self, state: dict[str, Any], *, tool_count: int, max_tool_calls: int
    ) -> str | None:
        """Return a rejection reason when the active source objective cannot finish."""
        ...

    def record_finish(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return source state prepared for the next fresh cycle."""
        ...


@runtime_checkable
class ApprovalAwareDataSource(Protocol):
    """Identify source actions that need a human decision before execution."""

    def approval_request(
        self, action: SourceToolCall, state: dict[str, Any]
    ) -> ApprovalRequest | None:
        """Return approval copy for a consequential action, when required."""
        ...
