"""Node behavior for the autonomous LangGraph supervisor."""

from collections.abc import Callable
from typing import Any, Literal
from uuid import uuid4

from langgraph.types import Command, interrupt

from eventpilot.core.agent_reasoning import (
    AgentTurn,
    AutonomousReasoningEngine,
    FinishCycle,
    SelectObjective,
    SendAlert,
    Wait,
    available_tool_types,
    parse_core_tool,
)
from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.autonomous.state import AutonomousAgentState, Sleep, ToolRoute
from eventpilot.core.monitoring import (
    after_wait,
    apply_execution,
    available_source_tools,
    initial_state,
    record_alert,
    record_finish,
    record_rejected_action,
    select_objective,
    should_continue_after_alert,
    validate_alert,
    validate_finish,
    validate_source_action,
    validate_wait,
)
from eventpilot.core.notifications import Notification, NotificationPriority
from eventpilot.core.reporting import (
    AgentDecisionEvent,
    AgentReporter,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
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


class AutonomousGraphNodes:
    """Implement graph nodes using explicitly injected runtime dependencies."""

    def __init__(
        self,
        agent: AutonomousReasoningEngine,
        source: DataSource,
        sink: NotificationSink,
        *,
        destination: str,
        sleep: Sleep,
        idle_sleep: Sleep | None,
        clock: Callable[[], float],
        max_wait_seconds: int | None,
        max_tool_calls_per_cycle: int,
        reporter: AgentReporter,
    ) -> None:
        """Bind node behavior to one graph's services and runtime policy."""
        self.agent = agent
        self.source = source
        self.sink = sink
        self.destination = destination
        self.sleep = sleep
        self.idle_sleep = idle_sleep
        self.clock = clock
        self.max_wait_seconds = max_wait_seconds
        self.max_tool_calls_per_cycle = max_tool_calls_per_cycle
        self.reporter = reporter

    def source_state(self, state: AutonomousAgentState) -> dict[str, Any]:
        """Return persisted graph-owned monitoring state or a fresh state."""
        return state.get("source_state", initial_state())

    async def reason(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Let the reasoning engine select one core or source-provided tool."""
        source_state = self.source_state(state)
        turn = await self.agent.decide(state.get("transcript", []), source_state)
        self.reporter.emit(
            AgentDecisionEvent(
                data_source=self.source.name,
                cycle_count=state.get("cycle_count", 0),
                tool_count=state.get("tool_count", 0),
                rationale=turn.rationale,
                tool=turn.action.tool_name,
                action_model=type(turn.action).__name__,
                arguments=turn.action.model_dump(mode="json", exclude={"tool"}),
                available_tools=sorted(
                    tool_type.model_fields["tool"].default
                    for tool_type in available_tool_types(
                        self.source,
                        source_state,
                        tool_count=state.get("tool_count", 0),
                        max_tool_calls=self.max_tool_calls_per_cycle,
                    )
                ),
                source_state=source_state,
            )
        )
        return {"turn": turn.model_dump(mode="json"), "outcome": "tool_selected"}

    def route_tool(self, state: AutonomousAgentState) -> ToolRoute:
        """Route the selected core or source tool to its graph node."""
        action = _turn(state, self.source).action
        if isinstance(action, SendAlert):
            return "send_alert"
        if isinstance(action, Wait):
            return "wait"
        if isinstance(action, FinishCycle):
            return "finish_cycle"
        if isinstance(action, SelectObjective):
            return "select_objective"
        source_state = self.source_state(state)
        if action.tool_name in available_source_tools(self.source, source_state):
            if validate_source_action(action, source_state):
                return "reject_action"
            if (
                isinstance(self.source, ApprovalAwareDataSource)
                and self.source.approval_request(action, source_state) is not None
            ):
                return "request_approval"
            return "source_tool"
        return "reject_action"

    async def execute_source_tool(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute one plugin-owned typed tool and reduce its normalized effects."""
        action = _turn(state, self.source).action
        source_state = self.source_state(state)
        execution = await self.source.execute(
            action,
            SourceContext(
                state=source_state,
                transcript=state.get("transcript", []),
                clock=self.clock,
            ),
        )
        result, next_source_state = apply_execution(
            execution, source_state, observed_at=self.clock()
        )
        update = _tool_result(state, action, result)
        update.update(
            source_state=next_source_state,
            pending_approval=None,
            approval_decision=None,
        )
        self.report_tool_result(state, action, result, update)
        return update

    def choose_objective(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Validate and persist a generic monitoring objective in graph state."""
        action = _expect_action(state, self.source, SelectObjective)
        result, next_source_state = select_objective(action, self.source_state(state))
        update = _tool_result(state, action, result)
        update["source_state"] = next_source_state
        self.report_tool_result(state, action, result, update)
        return update

    async def request_approval(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Deliver an approval request before the graph enters its interrupt node."""
        turn = _turn(state, self.source)
        action = turn.action
        if not isinstance(self.source, ApprovalAwareDataSource):
            raise TypeError(f"Source {self.source.name} does not define approval policy")
        requirement = self.source.approval_request(action, self.source_state(state))
        if requirement is None:
            raise ValueError(f"Tool {action.tool_name} does not require approval")
        arguments = action.model_dump(mode="json", exclude={"tool"})
        approval_id = str(uuid4())
        receipt = await self.sink.send(
            self.destination,
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
        self.reporter.emit(
            ApprovalRequestedEvent(
                data_source=self.source.name,
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
        self, state: AutonomousAgentState
    ) -> Command[Literal["source_tool", "reject_approval"]]:
        """Pause durably and route the resumed operator decision with LangGraph."""
        pending = state.get("pending_approval")
        if pending is None:
            raise ValueError("Approval interrupt requires a pending action")
        decision = ApprovalDecision(interrupt(pending))
        self.reporter.emit(
            ApprovalResolvedEvent(
                data_source=self.source.name,
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
            goto=("source_tool" if decision is ApprovalDecision.APPROVED else "reject_approval"),
        )

    def reject_approval(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Record an operator rejection without executing the suspended source tool."""
        action = _turn(state, self.source).action
        update = _rejected_tool_result(state, action, "The operator rejected this action.")
        update["source_state"] = record_rejected_action(action, self.source_state(state))
        update["pending_approval"] = None
        update["approval_decision"] = ApprovalDecision.REJECTED.value
        self.report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
        return update

    async def send_alert(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Deliver an alert and continue useful work within the current cycle."""
        action = _expect_action(state, self.source, SendAlert)
        source_state = self.source_state(state)
        rejection = validate_alert(action.resource_ids, source_state)
        if rejection:
            update = _rejected_tool_result(state, action, rejection)
            self.report_tool_result(
                state, action, update.get("transcript", [])[-1]["result"], update
            )
            return update
        receipt = await self.sink.send(
            self.destination,
            Notification(title=action.title, body=action.body, priority=action.priority),
        )
        result = receipt.model_dump(mode="json")
        update = _tool_result(state, action, result)
        update["source_state"] = record_alert(
            action.resource_ids, source_state, delivered_at=self.clock()
        )
        summary = f"Delivered alert for {', '.join(action.resource_ids)}."
        if not should_continue_after_alert(update["source_state"]):
            update.update(
                cycle_summary=summary,
                cycle_count=state.get("cycle_count", 0) + 1,
                tool_count=0,
                outcome="cycle_finished",
            )
        self.report_tool_result(state, action, result, update)
        if update.get("outcome") == "cycle_finished":
            self.report_cycle_finished(update, summary)
        return update

    async def wait(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Pause for the model-selected interval and advance graph scheduling state."""
        action = _expect_action(state, self.source, Wait)
        source_state = self.source_state(state)
        rejection = validate_wait(source_state)
        if rejection:
            update = _rejected_tool_result(state, action, rejection)
            self.report_tool_result(
                state, action, update.get("transcript", [])[-1]["result"], update
            )
            return update
        elapsed_seconds = min(action.seconds, self.max_wait_seconds or action.seconds)
        wake_at = self.clock() + elapsed_seconds
        sleep = (
            self.idle_sleep
            if source_state.get("phase") == "idle" and self.idle_sleep
            else self.sleep
        )
        await sleep(elapsed_seconds)
        result = {
            "status": "completed",
            "requested_seconds": action.seconds,
            "elapsed_seconds": elapsed_seconds,
            "reason": action.reason,
        }
        update = _tool_result(state, action, result)
        update["source_state"] = after_wait(
            source_state, requested_seconds=action.seconds, wake_at=wake_at
        )
        self.report_tool_result(
            state,
            action,
            result,
            update,
            requested_wait_seconds=action.seconds,
            elapsed_wait_seconds=elapsed_seconds,
        )
        return update

    def finish_cycle(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """End one bounded invocation after graph-owned policy approves completion."""
        action = _expect_action(state, self.source, FinishCycle)
        rejection = validate_finish(self.source_state(state))
        if rejection:
            update = _rejected_tool_result(state, action, rejection)
            self.report_tool_result(
                state, action, update.get("transcript", [])[-1]["result"], update
            )
            return update
        update: AutonomousAgentState = {
            "cycle_summary": action.summary,
            "cycle_count": state.get("cycle_count", 0) + 1,
            "source_state": record_finish(self.source_state(state)),
            "tool_count": 0,
            "outcome": "cycle_finished",
        }
        self.report_cycle_finished(update, action.summary)
        return update

    def reject_action(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Reject an unavailable plugin tool without executing side effects."""
        action = _turn(state, self.source).action
        update = _rejected_tool_result(
            state,
            action,
            f"Tool {action.tool_name} is unavailable in the current {self.source.name} state.",
        )
        self.report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
        return update

    @staticmethod
    def route_cycle_end(state: AutonomousAgentState) -> Literal["end", "agent"]:
        """End a completed cycle or return a rejected attempt to the agent."""
        return "end" if state.get("outcome") == "cycle_finished" else "agent"

    def report_tool_result(
        self,
        previous: AutonomousAgentState,
        action: SourceToolCall,
        result: dict[str, Any],
        update: AutonomousAgentState,
        *,
        requested_wait_seconds: int | None = None,
        elapsed_wait_seconds: float | None = None,
    ) -> None:
        """Emit one tool execution with its arguments, result, and next source state."""
        self.reporter.emit(
            ToolResultEvent(
                data_source=self.source.name,
                cycle_count=update.get("cycle_count", previous.get("cycle_count", 0)),
                tool_count=update.get("tool_count", previous.get("tool_count", 0)),
                tool=action.tool_name,
                action_model=type(action).__name__,
                arguments=action.model_dump(mode="json", exclude={"tool"}),
                result=result,
                outcome=update.get("outcome", "unknown"),
                source_state=update.get("source_state", self.source_state(previous)),
                requested_wait_seconds=requested_wait_seconds,
                elapsed_wait_seconds=elapsed_wait_seconds,
            )
        )

    def report_cycle_finished(self, state: AutonomousAgentState, summary: str) -> None:
        """Emit the durable state returned at the end of a finite cycle."""
        self.reporter.emit(
            CycleFinishedEvent(
                data_source=self.source.name,
                cycle_count=state.get("cycle_count", 0),
                tool_count=state.get("tool_count", 0),
                summary=summary,
                outcome=state.get("outcome", "cycle_finished"),
                source_state=self.source_state(state),
            )
        )


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
