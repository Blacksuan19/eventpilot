"""Continuous runtime for finite autonomous graph cycles."""

import asyncio
from typing import Any

from langgraph.types import Command

from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.autonomous.state import AutonomousAgentState


class AgentRuntime:
    """Continuously start finite invocations on one durable supervisor thread."""

    thread_id = "eventpilot-supervisor"

    def __init__(
        self,
        graph: Any,
        *,
        recursion_limit: int = 10_000,
        automatic_approval: ApprovalDecision | None = None,
    ) -> None:
        """Bind the process loop to the global autonomous-agent thread."""
        self._graph = graph
        self._config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": recursion_limit,
        }
        self._automatic_approval = automatic_approval
        self._resume_queue: asyncio.Queue[Command[Any]] = asyncio.Queue()
        self._resume_lock = asyncio.Lock()
        self._submitted_approval_ids: set[str] = set()

    async def run(self, *, max_cycles: int | None = None) -> AutonomousAgentState:
        """Run cycles while preserving and resuming native LangGraph interrupts."""
        completed = 0
        result: dict[str, Any] = {}
        graph_input: dict[str, Any] | Command[Any] = await self._initial_input()
        while max_cycles is None or completed < max_cycles:
            result = await self._graph.ainvoke(graph_input, config=self._config)
            if result.get("__interrupt__"):
                pending = result.get("pending_approval")
                approval_id = str(pending["id"]) if isinstance(pending, dict) else None
                graph_input = await self._resume_input()
                if approval_id:
                    self._submitted_approval_ids.discard(approval_id)
                continue
            completed += 1
            graph_input = {"transcript": []}
        return AutonomousAgentState(**result)

    async def resolve_approval(self, approval_id: str, decision: ApprovalDecision) -> bool:
        """Resume the current LangGraph interrupt with a validated operator decision."""
        async with self._resume_lock:
            if approval_id in self._submitted_approval_ids:
                return False
            pending = None
            for _ in range(50):
                pending = await self._pending_approval(require_interrupt=False)
                if pending is not None:
                    break
                await asyncio.sleep(0.01)
            if pending is None or pending.get("id") != approval_id:
                return False
            self._submitted_approval_ids.add(approval_id)
            self._resume_queue.put_nowait(Command(resume=decision.value))
            return True

    async def _initial_input(self) -> dict[str, Any] | Command[Any]:
        """Resume a checkpointed interrupt or start a fresh finite cycle."""
        pending = await self._pending_approval()
        if pending is None:
            return {"transcript": []}
        return await self._resume_input()

    async def _resume_input(self) -> Command[Any]:
        """Return an automatic or externally supplied interrupt-resume command."""
        if self._automatic_approval is not None:
            return Command(resume=self._automatic_approval.value)
        return await self._resume_queue.get()

    async def _pending_approval(self, *, require_interrupt: bool = True) -> dict[str, Any] | None:
        """Read a durable pending approval from the graph's current checkpoint."""
        snapshot = await self._graph.aget_state(self._config)
        if require_interrupt and not any(task.interrupts for task in snapshot.tasks):
            return None
        pending = snapshot.values.get("pending_approval")
        return dict(pending) if isinstance(pending, dict) else None
