"""Continuous runtime for persisted autonomous graph invocations."""

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
        recursion_limit: int = 256,
    ) -> None:
        """Bind the process loop to the global autonomous-agent thread."""
        self._graph = graph
        self._config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": recursion_limit,
        }
        self._resume_queue: asyncio.Queue[tuple[str, Command[Any]]] = asyncio.Queue()
        self._resume_lock = asyncio.Lock()
        self._submitted_approval_ids: set[str] = set()
        self._approval_ready = asyncio.Event()
        self._ready_approval_id: str | None = None

    async def run(self, *, max_invocations: int | None = None) -> AutonomousAgentState:
        """Run bounded invocations while preserving native LangGraph interrupts."""
        completed = 0
        result: dict[str, Any] = {}
        graph_input, resuming_approval_id = await self._initial_input()
        while max_invocations is None or completed < max_invocations:
            result = await self._graph.ainvoke(graph_input, config=self._config)
            if resuming_approval_id is not None:
                self._submitted_approval_ids.discard(resuming_approval_id)
                resuming_approval_id = None
            if result.get("__interrupt__"):
                pending = result.get("pending_approval")
                approval_id = str(pending["id"]) if isinstance(pending, dict) else None
                if approval_id is None:
                    raise RuntimeError("Approval interrupt is missing its pending identifier")
                self._signal_approval_ready(approval_id)
                graph_input, resuming_approval_id = await self._resume_input(approval_id)
                continue
            completed += 1
            graph_input = {"transcript": [], "tool_count": 0}
        return AutonomousAgentState(**result)

    async def resolve_approval(self, approval_id: str, decision: ApprovalDecision) -> bool:
        """Resume the current LangGraph interrupt with a validated operator decision."""
        async with self._resume_lock:
            if approval_id in self._submitted_approval_ids:
                return False
            pending = await self._pending_approval()
            if pending is None or pending.get("id") != approval_id:
                return False
            await self._approval_ready.wait()
            if self._ready_approval_id != approval_id:
                return False
            self._submitted_approval_ids.add(approval_id)
            self._resume_queue.put_nowait((approval_id, Command(resume=decision.value)))
            return True

    async def _initial_input(
        self,
    ) -> tuple[dict[str, Any] | Command[Any] | None, str | None]:
        """Resume unfinished checkpoint work or start a fresh graph invocation."""
        snapshot = await self._graph.aget_state(self._config)
        if any(task.interrupts for task in snapshot.tasks):
            pending = snapshot.values.get("pending_approval")
            if not isinstance(pending, dict) or "id" not in pending:
                raise RuntimeError("Approval interrupt is missing its pending identifier")
            approval_id = str(pending["id"])
            self._signal_approval_ready(approval_id)
            return await self._resume_input(approval_id)
        if snapshot.next:
            return None, None
        return {"transcript": [], "tool_count": 0}, None

    async def _resume_input(self, approval_id: str) -> tuple[Command[Any], str]:
        """Claim and return the command for one exact pending approval."""
        submitted_id, command = await self._resume_queue.get()
        if submitted_id != approval_id:
            raise RuntimeError(
                f"Approval command {submitted_id} cannot resume pending approval {approval_id}"
            )
        self._approval_ready.clear()
        self._ready_approval_id = None
        return command, submitted_id

    async def _pending_approval(self) -> dict[str, Any] | None:
        """Read a checkpointed pending approval before awaiting interrupt readiness."""
        snapshot = await self._graph.aget_state(self._config)
        pending = snapshot.values.get("pending_approval")
        return dict(pending) if isinstance(pending, dict) else None

    def _signal_approval_ready(self, approval_id: str) -> None:
        """Publish the exact interrupt that can accept an operator decision."""
        self._ready_approval_id = approval_id
        self._approval_ready.set()
