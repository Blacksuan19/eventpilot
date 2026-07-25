"""Build and run the generic autonomous LangGraph tool loop."""

import asyncio
from collections.abc import Awaitable, Callable
from time import time
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from eventpilot.core.agent_reasoning import (
    AgentTurn,
    AutonomousReasoningEngine,
    FinishCycle,
    SendAlert,
    Wait,
    available_tool_types,
    parse_core_tool,
)
from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.notifications import Notification, NotificationPriority
from eventpilot.core.reporting import (
    AgentDecisionEvent,
    AgentReporter,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    ConsoleAgentReporter,
    CycleFinishedEvent,
    ToolResultEvent,
)
from eventpilot.notifications.base import NotificationSink
from eventpilot.sources.base import (
    ApprovalAwareDataSource,
    DataSource,
    SourceContext,
    SourceToolCall,
)


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


def build_autonomous_graph(
    agent: AutonomousReasoningEngine,
    source: DataSource,
    sink: NotificationSink,
    *,
    destination: str = "local-console",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    sleep: Sleep = asyncio.sleep,
    idle_sleep: Sleep | None = None,
    clock: Callable[[], float] = time,
    max_wait_seconds: int | None = None,
    max_tool_calls_per_cycle: int = 32,
    reporter: AgentReporter | None = None,
) -> Any:
    """Build a generic supervisor around a registered monitoring data source."""
    event_reporter = reporter or ConsoleAgentReporter()

    def source_state(state: AutonomousAgentState) -> dict[str, Any]:
        """Return persisted source state or the plugin's initial state."""
        return state.get("source_state", source.initial_state())

    async def reason(state: AutonomousAgentState) -> AutonomousAgentState:
        """Let the reasoning engine select one core or source-provided tool."""
        turn = await agent.decide(state.get("transcript", []), source_state(state))
        event_reporter.emit(
            AgentDecisionEvent(
                data_source=source.name,
                cycle_count=state.get("cycle_count", 0),
                tool_count=state.get("tool_count", 0),
                rationale=turn.rationale,
                tool=turn.action.tool_name,
                action_model=type(turn.action).__name__,
                arguments=turn.action.model_dump(mode="json", exclude={"tool"}),
                available_tools=sorted(
                    tool_type.model_fields["tool"].default
                    for tool_type in available_tool_types(
                        source,
                        source_state(state),
                        tool_count=state.get("tool_count", 0),
                        max_tool_calls=max_tool_calls_per_cycle,
                    )
                ),
                source_state=source_state(state),
            )
        )
        return {"turn": turn.model_dump(mode="json"), "outcome": "tool_selected"}

    def route_tool(state: AutonomousAgentState) -> str:
        """Route core tools and source tools allowed by deterministic plugin policy."""
        action = _turn(state, source).action
        if isinstance(action, SendAlert):
            return "send_alert"
        if isinstance(action, Wait):
            return "wait"
        if isinstance(action, FinishCycle):
            return "finish_cycle"
        if action.tool_name in source.available_tools(source_state(state)):
            if (
                isinstance(source, ApprovalAwareDataSource)
                and source.approval_request(action, source_state(state)) is not None
            ):
                return "request_approval"
            return "source_tool"
        return "reject_action"

    async def source_tool(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute one plugin-owned typed tool without knowing platform semantics."""
        action = _turn(state, source).action
        context = SourceContext(
            state=source_state(state),
            transcript=state.get("transcript", []),
            clock=clock,
            max_tool_calls_per_cycle=max_tool_calls_per_cycle,
        )
        execution = await source.execute(action, context)
        update = _tool_result(state, action, execution.result)
        update["source_state"] = execution.state
        update["pending_approval"] = None
        update["approval_decision"] = None
        report_tool_result(state, action, execution.result, update)
        return update

    async def request_approval(state: AutonomousAgentState) -> AutonomousAgentState:
        """Deliver an approval request before the graph enters its interrupt node."""
        turn = _turn(state, source)
        action = turn.action
        if not isinstance(source, ApprovalAwareDataSource):
            raise TypeError(f"Source {source.name} does not define approval policy")
        requirement = source.approval_request(action, source_state(state))
        if requirement is None:
            raise ValueError(f"Tool {action.tool_name} does not require approval")
        arguments = action.model_dump(mode="json", exclude={"tool"})
        approval_id = str(uuid4())
        receipt = await sink.send(
            destination,
            Notification(
                title=requirement.title,
                body=requirement.body,
                priority=NotificationPriority.HIGH,
            ),
        )
        pending = {
            "id": approval_id,
            "title": requirement.title,
            "body": requirement.body,
            "rationale": turn.rationale,
            "resource_ids": list(requirement.resource_ids),
            "tool": action.tool_name,
            "action_model": type(action).__name__,
            "arguments": arguments,
            "delivery": receipt.model_dump(mode="json"),
        }
        event_reporter.emit(
            ApprovalRequestedEvent(
                data_source=source.name,
                cycle_count=state.get("cycle_count", 0),
                tool_count=state.get("tool_count", 0),
                approval_id=approval_id,
                title=requirement.title,
                body=requirement.body,
                resource_ids=list(requirement.resource_ids),
                tool=action.tool_name,
                action_model=type(action).__name__,
                arguments=arguments,
                delivery=receipt.model_dump(mode="json"),
            )
        )
        return {"pending_approval": pending, "approval_decision": None}

    def human_approval(
        state: AutonomousAgentState,
    ) -> Command[Literal["source_tool", "reject_approval"]]:
        """Pause durably and route the resumed operator decision with LangGraph."""
        pending = state.get("pending_approval")
        if pending is None:
            raise ValueError("Approval interrupt requires a pending action")
        decision = ApprovalDecision(interrupt(pending))
        event_reporter.emit(
            ApprovalResolvedEvent(
                data_source=source.name,
                cycle_count=state.get("cycle_count", 0),
                tool_count=state.get("tool_count", 0),
                approval_id=str(pending["id"]),
                decision=decision.value,
                tool=str(pending["tool"]),
                action_model=str(pending["action_model"]),
                arguments=dict(pending["arguments"]),
            )
        )
        return Command(
            update={"approval_decision": decision.value},
            goto="source_tool" if decision is ApprovalDecision.APPROVED else "reject_approval",
        )

    async def reject_approval(state: AutonomousAgentState) -> AutonomousAgentState:
        """Record an operator rejection without executing the suspended source tool."""
        action = _turn(state, source).action
        update = _rejected_tool_result(state, action, "The operator rejected this action.")
        update["pending_approval"] = None
        update["approval_decision"] = ApprovalDecision.REJECTED.value
        report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
        return update

    async def send_alert(state: AutonomousAgentState) -> AutonomousAgentState:
        """Deliver an alert and continue working within the current cycle."""
        action = _expect_action(state, source, SendAlert)
        current_source_state = source_state(state)
        rejection = source.validate_alert(action.resource_ids, current_source_state)
        if rejection:
            update = _rejected_tool_result(state, action, rejection)
            report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
            return update
        receipt = await sink.send(
            destination,
            Notification(title=action.title, body=action.body, priority=action.priority),
        )
        update = _tool_result(state, action, receipt.model_dump(mode="json"))
        update["source_state"] = source.record_alert(
            action.resource_ids, current_source_state, delivered_at=clock()
        )
        if not source.should_continue_after_alert(update["source_state"]):
            update.update(
                cycle_summary=f"Delivered alert for {', '.join(action.resource_ids)}.",
                cycle_count=state.get("cycle_count", 0) + 1,
                tool_count=0,
                outcome="cycle_finished",
            )
        report_tool_result(state, action, receipt.model_dump(mode="json"), update)
        if update.get("outcome") == "cycle_finished":
            report_cycle_finished(update, f"Delivered alert for {', '.join(action.resource_ids)}.")
        return update

    async def wait(state: AutonomousAgentState) -> AutonomousAgentState:
        """Pause for the model-selected interval and notify the source scheduler."""
        action = _expect_action(state, source, Wait)
        rejection = source.validate_wait(source_state(state))
        if rejection:
            update = _rejected_tool_result(state, action, rejection)
            report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
            return update
        elapsed_seconds = min(action.seconds, max_wait_seconds or action.seconds)
        wake_at = clock() + elapsed_seconds
        wait_sleep = idle_sleep if source.is_idle(source_state(state)) and idle_sleep else sleep
        await wait_sleep(elapsed_seconds)
        update = _tool_result(
            state,
            action,
            {
                "status": "completed",
                "requested_seconds": action.seconds,
                "elapsed_seconds": elapsed_seconds,
                "reason": action.reason,
            },
        )
        update["source_state"] = source.after_wait(
            source_state(state), requested_seconds=action.seconds, wake_at=wake_at
        )
        report_tool_result(
            state,
            action,
            update.get("transcript", [])[-1]["result"],
            update,
            requested_wait_seconds=action.seconds,
            elapsed_wait_seconds=elapsed_seconds,
        )
        return update

    async def finish_cycle(state: AutonomousAgentState) -> AutonomousAgentState:
        """End one bounded invocation after source-owned policy approves completion."""
        action = _expect_action(state, source, FinishCycle)
        rejection = source.validate_finish(
            source_state(state),
            tool_count=state.get("tool_count", 0),
            max_tool_calls=max_tool_calls_per_cycle,
        )
        if rejection:
            update = _rejected_tool_result(state, action, rejection)
            report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
            return update
        update: AutonomousAgentState = {
            "cycle_summary": action.summary,
            "cycle_count": state.get("cycle_count", 0) + 1,
            "source_state": source.record_finish(source_state(state)),
            "tool_count": 0,
            "outcome": "cycle_finished",
        }
        report_cycle_finished(update, action.summary)
        return update

    async def reject_action(state: AutonomousAgentState) -> AutonomousAgentState:
        """Reject an unavailable plugin tool without executing side effects."""
        action = _turn(state, source).action
        update = _rejected_tool_result(
            state,
            action,
            f"Tool {action.tool_name} is unavailable in the current {source.name} state.",
        )
        report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
        return update

    def report_tool_result(
        previous: AutonomousAgentState,
        action: SourceToolCall,
        result: dict[str, Any],
        update: AutonomousAgentState,
        *,
        requested_wait_seconds: int | None = None,
        elapsed_wait_seconds: float | None = None,
    ) -> None:
        """Emit one tool execution with its arguments, result, and next source state."""
        event_reporter.emit(
            ToolResultEvent(
                data_source=source.name,
                cycle_count=update.get("cycle_count", previous.get("cycle_count", 0)),
                tool_count=update.get("tool_count", previous.get("tool_count", 0)),
                tool=action.tool_name,
                action_model=type(action).__name__,
                arguments=action.model_dump(mode="json", exclude={"tool"}),
                result=result,
                outcome=update.get("outcome", "unknown"),
                source_state=update.get("source_state", source_state(previous)),
                requested_wait_seconds=requested_wait_seconds,
                elapsed_wait_seconds=elapsed_wait_seconds,
            )
        )

    def report_cycle_finished(state: AutonomousAgentState, summary: str) -> None:
        """Emit the durable state returned at the end of a finite cycle."""
        event_reporter.emit(
            CycleFinishedEvent(
                data_source=source.name,
                cycle_count=state.get("cycle_count", 0),
                tool_count=state.get("tool_count", 0),
                summary=summary,
                outcome=state.get("outcome", "cycle_finished"),
                source_state=source_state(state),
            )
        )

    builder = StateGraph(AutonomousAgentState)
    builder.add_node("agent", reason)
    builder.add_node("source_tool", source_tool)
    builder.add_node("request_approval", request_approval)
    builder.add_node("human_approval", human_approval)
    builder.add_node("reject_approval", reject_approval)
    builder.add_node("send_alert", send_alert)
    builder.add_node("wait", wait)
    builder.add_node("finish_cycle", finish_cycle)
    builder.add_node("reject_action", reject_action)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_tool,
        {
            "source_tool": "source_tool",
            "request_approval": "request_approval",
            "send_alert": "send_alert",
            "wait": "wait",
            "finish_cycle": "finish_cycle",
            "reject_action": "reject_action",
        },
    )
    builder.add_edge("request_approval", "human_approval")
    builder.add_edge("source_tool", "agent")
    builder.add_edge("reject_approval", "agent")
    builder.add_edge("wait", "agent")
    builder.add_edge("reject_action", "agent")
    builder.add_conditional_edges(
        "send_alert",
        lambda state: "end" if state.get("outcome") == "cycle_finished" else "agent",
        {"end": END, "agent": "agent"},
    )
    builder.add_conditional_edges(
        "finish_cycle",
        lambda state: "end" if state.get("outcome") == "cycle_finished" else "agent",
        {"end": END, "agent": "agent"},
    )
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def _turn(state: AutonomousAgentState, source: DataSource) -> AgentTurn:
    """Validate and reconstruct the latest dynamically typed agent turn."""
    raw_turn = state.get("turn")
    if raw_turn is None:
        raise ValueError("Tool execution requires an agent turn")
    payload = raw_turn["action"]
    action = parse_core_tool(payload) or source.parse_tool(payload)
    return AgentTurn(rationale=raw_turn["rationale"], action=action)


def _expect_action[ActionT: SourceToolCall](
    state: AutonomousAgentState, source: DataSource, kind: type[ActionT]
) -> ActionT:
    """Return a chosen action after verifying its dynamically routed type."""
    action = _turn(state, source).action
    if not isinstance(action, kind):
        raise TypeError(f"Expected {kind.__name__}, received {type(action).__name__}")
    return action


def _tool_result(
    state: AutonomousAgentState, action: SourceToolCall, result: dict[str, Any]
) -> AutonomousAgentState:
    """Append an executed core or source tool call to the working transcript."""
    transcript = [
        *state.get("transcript", []),
        {"tool": action.tool_name, "call": action.model_dump(mode="json"), "result": result},
    ]
    return {
        "transcript": transcript,
        "tool_count": state.get("tool_count", 0) + 1,
        "outcome": f"{action.tool_name}_completed",
    }


def _rejected_tool_result(
    state: AutonomousAgentState, action: SourceToolCall, reason: str
) -> AutonomousAgentState:
    """Record a deterministic policy rejection without executing side effects."""
    return _tool_result(state, action, {"status": "rejected", "reason": reason})


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
