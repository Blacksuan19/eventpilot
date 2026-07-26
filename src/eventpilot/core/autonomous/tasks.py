"""Checkpoint external side effects executed inside autonomous graph nodes."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.func import task

from eventpilot.core.notifications import DeliveryResult, Notification
from eventpilot.notifications.base import NotificationSink
from eventpilot.sources.base import (
    DataSource,
    SourceContext,
    SourceExecution,
    SourceToolCall,
    parse_source_execution,
    serialize_source_execution,
)


class _ExecuteSourceOperation:
    """Reconstruct and execute one source operation inside a LangGraph task."""

    def __init__(self, source: DataSource, clock: Callable[[], float]) -> None:
        """Capture runtime services that must stay outside checkpointed task inputs."""
        self._source = source
        self._clock = clock

    async def __call__(
        self,
        action_payload: dict[str, Any],
        source_state: dict[str, Any],
        transcript: list[dict[str, Any]],
        operation_id: str,
    ) -> dict[str, Any]:
        """Execute the reconstructed source tool and serialize its durable result."""
        action = self._source.parse_tool(action_payload)
        execution = await self._source.execute(
            action,
            SourceContext(
                state=source_state,
                transcript=transcript,
                clock=self._clock,
                operation_id=operation_id,
            ),
        )
        return {
            "execution": serialize_source_execution(execution),
            "observed_at": self._clock(),
        }


class _DeliverNotification:
    """Deliver one notification inside a checkpointed LangGraph task."""

    def __init__(self, sink: NotificationSink) -> None:
        """Capture the configured provider without persisting it as task input."""
        self._sink = sink

    async def __call__(
        self,
        destination: str,
        notification_payload: dict[str, Any],
        operation_id: str,
    ) -> dict[str, Any]:
        """Send a validated notification with a stable provider idempotency key."""
        receipt = await self._sink.send(
            destination,
            Notification.model_validate(notification_payload),
            idempotency_key=operation_id,
        )
        return receipt.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class DurableSourceResult:
    """Pair a restored source result with its original observation time."""

    execution: SourceExecution
    observed_at: float


class DurableOperations:
    """Expose external operations as replay-aware LangGraph tasks."""

    def __init__(
        self,
        source: DataSource,
        sink: NotificationSink,
        clock: Callable[[], float],
    ) -> None:
        """Create stable task definitions around injected runtime adapters."""
        self._execute_source = task(name="execute_source_operation")(
            _ExecuteSourceOperation(source, clock)
        )
        self._deliver_notification = task(name="deliver_notification")(_DeliverNotification(sink))

    async def execute_source(
        self,
        action: SourceToolCall,
        context: SourceContext,
        *,
        operation_id: str,
    ) -> DurableSourceResult:
        """Return a source result restored from its task checkpoint when available."""
        payload = await self._execute_source(
            action.model_dump(mode="json"),
            context.state,
            context.transcript,
            operation_id,
        )
        return DurableSourceResult(
            execution=parse_source_execution(payload["execution"]),
            observed_at=float(payload["observed_at"]),
        )

    async def deliver_notification(
        self,
        destination: str,
        notification: Notification,
        *,
        operation_id: str,
    ) -> DeliveryResult:
        """Return a delivery receipt restored from its task checkpoint when available."""
        payload = await self._deliver_notification(
            destination,
            notification.model_dump(mode="json"),
            operation_id,
        )
        return DeliveryResult.model_validate(payload)
