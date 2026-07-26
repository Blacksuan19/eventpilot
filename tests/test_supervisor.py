"""Verify dashboard agent task lifecycle supervision."""

import asyncio

from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.autonomous import AutonomousAgentState
from eventpilot.dashboard.supervisor import AgentHealth, AgentTaskSupervisor


class ControlledRuntime:
    """Expose a controllable run coroutine for supervisor tests."""

    def __init__(
        self,
        result: AutonomousAgentState | None = None,
        error: Exception | None = None,
    ) -> None:
        """Configure the runtime to block, return, or fail when released."""
        self.release = asyncio.Event()
        self.result = result
        self.error = error

    async def run(self) -> AutonomousAgentState:
        """Wait for release before returning or raising the configured outcome."""
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result or AutonomousAgentState()

    async def resolve_approval(self, approval_id: str, decision: ApprovalDecision) -> bool:
        """Reject approvals because this runtime only exercises task lifecycle."""
        return False


async def test_supervisor_reports_runtime_failure() -> None:
    """Expose a background task exception instead of leaving stale running health."""
    runtime = ControlledRuntime(error=RuntimeError("model unavailable"))

    async def reset_state() -> None:
        """Provide the reset hook required by the supervisor."""

    supervisor = AgentTaskSupervisor(lambda: runtime, reset_state)
    supervisor.start()
    assert supervisor.health().state == "running"

    runtime.release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert supervisor.health() == AgentHealth("failed", "RuntimeError: model unavailable")


async def test_supervisor_reset_replaces_a_stopped_task() -> None:
    """Clear durable state and return to running health after a reset."""
    runtimes = [ControlledRuntime(), ControlledRuntime()]
    reset_calls = 0

    async def reset_state() -> None:
        """Record durable reset execution."""
        nonlocal reset_calls
        reset_calls += 1

    supervisor = AgentTaskSupervisor(lambda: runtimes.pop(0), reset_state)
    supervisor.start()

    await supervisor.reset()

    assert reset_calls == 1
    assert supervisor.health().state == "running"
    await supervisor.stop()
    assert supervisor.health().state == "stopped"
