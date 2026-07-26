"""Define the narrow contract implemented by platform data sources."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from eventpilot.core.approvals import ApprovalRequest


@dataclass(frozen=True, slots=True)
class ToolAvailability:
    """Declare generic evidence required before exposing a source tool."""

    statuses: tuple[str, ...] = ()
    evidence_keys: tuple[str, ...] = ()
    requirement_key: str | None = None


class SourceToolCall(BaseModel):
    """Provide the common discriminator required by every source tool."""

    availability: ClassVar[ToolAvailability | None] = None
    parallel_safe: ClassVar[bool] = False
    retry_safe: ClassVar[bool] = False

    @property
    def tool_name(self) -> str:
        """Return the validated discriminator declared by a concrete tool model."""
        return str(self.model_dump()["tool"])


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Normalize one platform resource for graph-owned monitoring state."""

    resource_id: str
    status: str
    results_status: str | None = None
    active: bool = True
    result_ready: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceEffect:
    """Describe facts produced by a source tool without mutating graph state."""

    kind: Literal["discovery", "observation"]
    resources: tuple[ResourceSnapshot, ...] = ()
    resource_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    inspected: bool = False
    result_ready: bool = False
    wait_blocker: str | None = None
    clear_wait_blocker: bool = False
    required_action: str | None = None
    clear_required_action: bool = False


@dataclass(frozen=True, slots=True)
class SourceContext:
    """Expose durable monitoring state and safe runtime services to a tool handler."""

    state: dict[str, Any]
    transcript: list[dict[str, Any]]
    clock: Callable[[], float]


@dataclass(frozen=True, slots=True)
class SourceExecution:
    """Return a JSON tool result and normalized facts for graph reducers."""

    result: dict[str, Any]
    effects: tuple[SourceEffect, ...] = ()


def serialize_source_execution(execution: SourceExecution) -> dict[str, Any]:
    """Convert a source execution into checkpoint-safe graph state."""
    return {
        "result": execution.result,
        "effects": [asdict(effect) for effect in execution.effects],
    }


def parse_source_execution(payload: Mapping[str, Any]) -> SourceExecution:
    """Reconstruct a source execution after parallel LangGraph fan-in."""
    effects = tuple(
        SourceEffect(
            kind=effect["kind"],
            resources=tuple(ResourceSnapshot(**resource) for resource in effect["resources"]),
            resource_id=effect["resource_id"],
            evidence=effect["evidence"],
            inspected=effect["inspected"],
            result_ready=effect["result_ready"],
            wait_blocker=effect["wait_blocker"],
            clear_wait_blocker=effect["clear_wait_blocker"],
            required_action=effect["required_action"],
            clear_required_action=effect["clear_required_action"],
        )
        for effect in payload.get("effects", [])
    )
    return SourceExecution(result=dict(payload["result"]), effects=effects)


class DataSource(Protocol):
    """Expose typed platform tools while leaving orchestration to LangGraph."""

    name: str
    instructions: str
    discovery_tool: str
    tool_types: tuple[type[SourceToolCall], ...]

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate a persisted tool payload against the source's tool catalog."""
        ...

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Execute one platform tool and describe its observable effects."""
        ...


@runtime_checkable
class ApprovalAwareDataSource(Protocol):
    """Identify source actions that need a human decision before execution."""

    def approval_request(
        self, action: SourceToolCall, state: dict[str, Any]
    ) -> ApprovalRequest | None:
        """Return approval copy for a consequential action, when required."""
        ...
