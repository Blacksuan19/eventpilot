"""Track the dashboard agent task and expose its lifecycle state."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol

from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.autonomous import AutonomousAgentState

logger = logging.getLogger(__name__)

AgentLifecycleState = Literal["starting", "running", "failed", "stopped"]
ResetState = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AgentHealth:
    """Describe the current state of the supervised agent task."""

    state: AgentLifecycleState
    detail: str | None = None


class SupervisedRuntime(Protocol):
    """Expose the runtime operations required by dashboard supervision."""

    async def run(self) -> AutonomousAgentState:
        """Run the agent until cancellation or an unexpected exit."""
        ...

    async def resolve_approval(self, approval_id: str, decision: ApprovalDecision) -> bool:
        """Resume one pending approval when it belongs to this runtime."""
        ...


RuntimeFactory = Callable[[], SupervisedRuntime]


class AgentTaskSupervisor:
    """Own one agent task and record completion or failure for the dashboard."""

    def __init__(self, runtime_factory: RuntimeFactory, reset_state: ResetState) -> None:
        """Bind task lifecycle management to runtime construction and state reset hooks."""
        self._runtime_factory = runtime_factory
        self._reset_state = reset_state
        self._runtime: SupervisedRuntime | None = None
        self._task: asyncio.Task[AutonomousAgentState] | None = None
        self._health = AgentHealth("starting")
        self._reset_lock = asyncio.Lock()

    def start(self) -> None:
        """Start a fresh runtime and observe how its task exits."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("The agent task is already running")
        try:
            self._runtime = self._runtime_factory()
        except Exception as exc:
            self._record_failure(exc)
            raise
        self._health = AgentHealth("running")
        self._task = asyncio.create_task(self._runtime.run(), name="eventpilot-agent")
        self._task.add_done_callback(self._observe_completion)

    async def reset(self) -> None:
        """Stop the current task, clear durable state, and start a fresh runtime."""
        async with self._reset_lock:
            await self.stop()
            try:
                await self._reset_state()
                self.start()
            except Exception as exc:
                self._record_failure(exc)
                raise

    async def stop(self) -> None:
        """Cancel the current task and leave an explicit stopped state."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if self._task is task:
            self._health = AgentHealth("stopped")

    async def resolve_approval(self, approval_id: str, decision: ApprovalDecision) -> bool:
        """Forward an operator decision only while the agent task is running."""
        if self._health.state != "running" or self._runtime is None:
            return False
        return await self._runtime.resolve_approval(approval_id, decision)

    def health(self) -> AgentHealth:
        """Return an immutable snapshot of the current task lifecycle."""
        return self._health

    def _observe_completion(self, task: asyncio.Task[AutonomousAgentState]) -> None:
        """Translate task completion into dashboard-visible lifecycle state."""
        if task is not self._task:
            return
        if task.cancelled():
            self._health = AgentHealth("stopped")
            return
        error = task.exception()
        if error is not None:
            self._record_failure(error)
            return
        self._health = AgentHealth("stopped", "Agent runtime exited unexpectedly.")

    def _record_failure(self, error: BaseException) -> None:
        """Record and log one runtime lifecycle failure."""
        detail = f"{type(error).__name__}: {error}"
        self._health = AgentHealth("failed", detail)
        logger.error(
            "EventPilot agent task failed",
            exc_info=(type(error), error, error.__traceback__),
        )
