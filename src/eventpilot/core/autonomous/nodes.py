"""Node behavior for the autonomous LangGraph supervisor."""

from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite, Send, interrupt

from eventpilot.core.agent_reasoning import (
    AgentTurn,
    AutonomousReasoningEngine,
    SelectObjective,
    SendAlert,
    Wait,
    available_tool_types,
    parse_core_tool,
)
from eventpilot.core.approvals import ApprovalDecision
from eventpilot.core.autonomous.state import AutonomousAgentState, Sleep, ToolRoute
from eventpilot.core.autonomous.tasks import DurableOperations
from eventpilot.core.monitoring import (
    after_wait,
    apply_execution,
    available_source_tools,
    initial_state,
    record_alert,
    record_rejected_action,
    select_objective,
    should_continue_after_alert,
    validate_alert,
    validate_source_action,
    validate_source_actions,
    validate_wait,
)
from eventpilot.core.notifications import Notification, NotificationPriority
from eventpilot.core.reporting import (
    AgentActionSelection,
    AgentDecisionEvent,
    AgentReporter,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    InvocationFinishedEvent,
    ToolResultEvent,
)
from eventpilot.notifications.base import NotificationSink
from eventpilot.sources.base import (
    ApprovalAwareDataSource,
    DataSource,
    SourceContext,
    SourceToolCall,
    parse_source_execution,
    serialize_source_execution,
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
        reporter: AgentReporter,
    ) -> None:
        """Bind node behavior to one graph's services and runtime policy."""
        self.agent = agent
        self.source = source
        self.destination = destination
        self.sleep = sleep
        self.idle_sleep = idle_sleep
        self.clock = clock
        self.max_wait_seconds = max_wait_seconds
        self.reporter = reporter
        self.operations = DurableOperations(source, sink, clock)

    def source_state(self, state: AutonomousAgentState) -> dict[str, Any]:
        """Return persisted graph-owned monitoring state or a fresh state."""
        return state.get("source_state", initial_state())

    @staticmethod
    def operation_id(runtime: Runtime[Any]) -> str:
        """Return the stable LangGraph task identity required by external operations."""
        if runtime.execution_info is None:
            raise RuntimeError("Durable operations require LangGraph execution metadata")
        return runtime.execution_info.task_id

    async def reason(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Let the reasoning engine select one control action or source-action batch."""
        source_state = self.source_state(state)
        turn = await self.agent.decide(state.get("transcript", []), source_state)
        self.reporter.emit(
            AgentDecisionEvent(
                data_source=self.source.name,
                invocation_count=state.get("invocation_count", 0),
                tool_count=state.get("tool_count", 0),
                rationale=turn.rationale,
                actions=[
                    AgentActionSelection(
                        tool=action.tool_name,
                        action_model=type(action).__name__,
                        arguments=action.model_dump(mode="json", exclude={"tool"}),
                    )
                    for action in turn.actions
                ],
                parallel=len(turn.actions) > 1,
                available_tools=sorted(
                    tool_type.model_fields["tool"].default
                    for tool_type in available_tool_types(self.source, source_state)
                ),
                source_state=source_state,
            )
        )
        return {"turn": turn.model_dump(mode="json"), "outcome": "tool_selected"}

    def route_tool(self, state: AutonomousAgentState) -> ToolRoute | list[Send]:
        """Route one action normally or fan independent source reads out with Send."""
        turn = _turn(state, self.source)
        if len(turn.actions) > 1:
            if self.parallel_batch_rejection(turn.actions, self.source_state(state)):
                return "reject_action"
            return [
                Send(
                    "parallel_source_tool",
                    {
                        "turn": {
                            "rationale": turn.rationale,
                            "actions": [action.model_dump(mode="json")],
                        },
                        "source_state": self.source_state(state),
                        "transcript": state.get("transcript", []),
                        "invocation_count": state.get("invocation_count", 0),
                        "tool_count": state.get("tool_count", 0),
                        "parallel_action_index": index,
                    },
                )
                for index, action in enumerate(turn.actions)
            ]
        action = _single_action(turn)
        if isinstance(action, SendAlert):
            return "send_alert"
        if isinstance(action, Wait):
            return "wait"
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
            if action.retry_safe:
                return "retryable_source_tool"
            return "source_tool"
        return "reject_action"

    def parallel_batch_rejection(
        self, actions: list[SourceToolCall], source_state: dict[str, Any]
    ) -> str | None:
        """Reject batches that cannot safely share one pre-execution state snapshot."""
        available = available_source_tools(self.source, source_state)
        if any(action.tool_name not in available for action in actions):
            return "Every parallel action must be an available source tool."
        if any(not action.parallel_safe for action in actions):
            return "Only source tools declared parallel-safe may share a turn."
        return validate_source_actions(actions, source_state)

    async def execute_source_tool(
        self, state: AutonomousAgentState, runtime: Runtime[Any]
    ) -> AutonomousAgentState:
        """Execute one plugin-owned typed tool and reduce its normalized effects."""
        action = _single_action(_turn(state, self.source))
        source_state = self.source_state(state)
        completed = await self.operations.execute_source(
            action,
            SourceContext(
                state=source_state,
                transcript=state.get("transcript", []),
                clock=self.clock,
            ),
            operation_id=f"{self.operation_id(runtime)}:source",
        )
        result, next_source_state = apply_execution(
            completed.execution,
            source_state,
            observed_at=completed.observed_at,
        )
        update = _tool_result(state, action, result)
        update.update(
            source_state=next_source_state,
            pending_approval=None,
            approval_decision=None,
        )
        self.report_tool_result(state, action, result, update)
        return update

    async def execute_parallel_source_tool(
        self, state: AutonomousAgentState, runtime: Runtime[Any]
    ) -> AutonomousAgentState:
        """Execute one read-only Send branch without mutating shared monitoring state."""
        action = _single_action(_turn(state, self.source))
        completed = await self.operations.execute_source(
            action,
            SourceContext(
                state=self.source_state(state),
                transcript=state.get("transcript", []),
                clock=self.clock,
            ),
            operation_id=f"{self.operation_id(runtime)}:source",
        )
        return {
            "parallel_results": [
                {
                    "index": state.get("parallel_action_index", 0),
                    "action": action.model_dump(mode="json"),
                    "execution": serialize_source_execution(completed.execution),
                    "observed_at": completed.observed_at,
                }
            ]
        }

    def reduce_parallel_source_tools(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Apply concurrent read results to durable state in original selection order."""
        working = AutonomousAgentState(**state)
        for item in sorted(state.get("parallel_results", []), key=lambda result: result["index"]):
            action = self.source.parse_tool(item["action"])
            execution = parse_source_execution(item["execution"])
            result, next_source_state = apply_execution(
                execution,
                self.source_state(working),
                observed_at=float(item["observed_at"]),
            )
            update = _tool_result(working, action, result)
            update.update(
                source_state=next_source_state,
                pending_approval=None,
                approval_decision=None,
            )
            self.report_tool_result(working, action, result, update)
            working.update(update)
        return {
            "transcript": working.get("transcript", []),
            "tool_count": working.get("tool_count", state.get("tool_count", 0)),
            "source_state": self.source_state(working),
            "pending_approval": None,
            "approval_decision": None,
            "outcome": "parallel_source_tools_completed",
            "parallel_results": cast(Any, Overwrite([])),
        }

    def choose_objective(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Validate and persist a generic monitoring objective in graph state."""
        action = _expect_action(state, self.source, SelectObjective)
        result, next_source_state = select_objective(action, self.source_state(state))
        update = _tool_result(state, action, result)
        update["source_state"] = next_source_state
        self.report_tool_result(state, action, result, update)
        return update

    async def request_approval(
        self, state: AutonomousAgentState, runtime: Runtime[Any]
    ) -> AutonomousAgentState:
        """Deliver an approval request before the graph enters its interrupt node."""
        turn = _turn(state, self.source)
        action = _single_action(turn)
        if not isinstance(self.source, ApprovalAwareDataSource):
            raise TypeError(f"Source {self.source.name} does not define approval policy")
        requirement = self.source.approval_request(action, self.source_state(state))
        if requirement is None:
            raise ValueError(f"Tool {action.tool_name} does not require approval")
        arguments = action.model_dump(mode="json", exclude={"tool"})
        operation_id = self.operation_id(runtime)
        approval_id = f"approval-{operation_id}"
        receipt = await self.operations.deliver_notification(
            self.destination,
            Notification(
                title=requirement.title,
                body=requirement.body,
                priority=NotificationPriority.HIGH,
            ),
            operation_id=f"{operation_id}:approval",
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
        return {"pending_approval": pending, "approval_decision": None}

    def publish_approval(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Expose an approval only after its pending action has been checkpointed."""
        pending = state.get("pending_approval")
        if not isinstance(pending, dict):
            raise ValueError("Approval publication requires a pending action")
        delivery = pending.get("delivery")
        if not isinstance(delivery, dict):
            raise ValueError("Approval publication requires a delivery receipt")
        self.reporter.emit(
            ApprovalRequestedEvent(
                data_source=self.source.name,
                invocation_count=state.get("invocation_count", 0),
                tool_count=state.get("tool_count", 0),
                approval_id=str(pending["id"]),
                title=str(pending["title"]),
                body=str(pending["body"]),
                resource_ids=[str(item) for item in pending["resource_ids"]],
                tool=str(pending["tool"]),
                action_model=str(pending["action_model"]),
                arguments=dict(pending["arguments"]),
                delivery=delivery,
            )
        )
        return {}

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
                invocation_count=state.get("invocation_count", 0),
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
        action = _single_action(_turn(state, self.source))
        update = _rejected_tool_result(state, action, "The operator rejected this action.")
        update["source_state"] = record_rejected_action(action, self.source_state(state))
        update["pending_approval"] = None
        update["approval_decision"] = ApprovalDecision.REJECTED.value
        self.report_tool_result(state, action, update.get("transcript", [])[-1]["result"], update)
        return update

    async def send_alert(
        self, state: AutonomousAgentState, runtime: Runtime[Any]
    ) -> AutonomousAgentState:
        """Deliver an alert and end when the current objective is complete."""
        action = _expect_action(state, self.source, SendAlert)
        source_state = self.source_state(state)
        rejection = validate_alert(action.resource_ids, source_state)
        if rejection:
            update = _rejected_tool_result(state, action, rejection)
            self.report_tool_result(
                state, action, update.get("transcript", [])[-1]["result"], update
            )
            return update
        receipt = await self.operations.deliver_notification(
            self.destination,
            Notification(title=action.title, body=action.body, priority=action.priority),
            operation_id=f"{self.operation_id(runtime)}:alert",
        )
        result = receipt.model_dump(mode="json")
        update = _tool_result(state, action, result)
        update["source_state"] = record_alert(
            action.resource_ids, source_state, delivered_at=self.clock()
        )
        summary = f"Delivered alert for {', '.join(action.resource_ids)}."
        if not should_continue_after_alert(update["source_state"]):
            update.update(
                invocation_summary=summary,
                outcome="invocation_ready_to_end",
            )
        self.report_tool_result(state, action, result, update)
        return update

    def prepare_wait(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Validate a wait and checkpoint its deadline before the process sleeps."""
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
        return {
            "pending_wait": {
                "requested_seconds": action.seconds,
                "elapsed_seconds": elapsed_seconds,
                "reason": action.reason,
                "wake_at": self.clock() + elapsed_seconds,
            }
        }

    @staticmethod
    def route_prepared_wait(state: AutonomousAgentState) -> Literal["wait", "agent"]:
        """Continue valid waits and return rejected waits to the agent."""
        return "wait" if state.get("pending_wait") else "agent"

    async def wait(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Sleep until the checkpointed deadline and advance scheduling state."""
        action = _expect_action(state, self.source, Wait)
        source_state = self.source_state(state)
        pending = state.get("pending_wait")
        if not isinstance(pending, dict):
            raise ValueError("Wait execution requires a checkpointed deadline")
        requested_seconds = int(pending["requested_seconds"])
        elapsed_seconds = float(pending["elapsed_seconds"])
        wake_at = float(pending["wake_at"])
        remaining_seconds = max(0.0, wake_at - self.clock())
        sleep = (
            self.idle_sleep
            if source_state.get("phase") == "idle" and self.idle_sleep
            else self.sleep
        )
        if remaining_seconds:
            await sleep(remaining_seconds)
        result = {
            "status": "completed",
            "requested_seconds": requested_seconds,
            "elapsed_seconds": elapsed_seconds,
            "reason": str(pending["reason"]),
        }
        update = _tool_result(state, action, result)
        update["source_state"] = after_wait(
            source_state, requested_seconds=requested_seconds, wake_at=wake_at
        )
        update["pending_wait"] = None
        unit = "second" if elapsed_seconds == 1 else "seconds"
        summary = f"Waited {elapsed_seconds:g} {unit} before the next invocation."
        update.update(
            invocation_summary=summary,
            outcome="invocation_ready_to_end",
        )
        self.report_tool_result(
            state,
            action,
            result,
            update,
            requested_wait_seconds=requested_seconds,
            elapsed_wait_seconds=elapsed_seconds,
        )
        return update

    def end_invocation(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Finalize a graph invocation at an explicit checkpointed topology boundary."""
        summary = state.get("invocation_summary")
        if summary is None:
            raise ValueError("Ending an invocation requires a summary")
        update: AutonomousAgentState = {
            "invocation_count": state.get("invocation_count", 0) + 1,
            "outcome": "invocation_finished",
        }
        completed = AutonomousAgentState(**state)
        completed.update(update)
        self.report_invocation_finished(completed, summary)
        return update

    def reject_action(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Reject unavailable or unsafe selected tools without executing side effects."""
        turn = _turn(state, self.source)
        reason = (
            self.parallel_batch_rejection(turn.actions, self.source_state(state))
            if len(turn.actions) > 1
            else None
        )
        working = AutonomousAgentState(**state)
        for action in turn.actions:
            action_reason = reason or (
                f"Tool {action.tool_name} is unavailable in the current {self.source.name} state."
            )
            update = _rejected_tool_result(working, action, action_reason)
            self.report_tool_result(
                working, action, update.get("transcript", [])[-1]["result"], update
            )
            working.update(update)
        return {
            "transcript": working.get("transcript", []),
            "tool_count": working.get("tool_count", state.get("tool_count", 0)),
            "outcome": working.get("outcome", "tools_rejected"),
        }

    @staticmethod
    def route_invocation_end(state: AutonomousAgentState) -> Literal["end", "agent"]:
        """End completed work or return a rejected attempt to the agent."""
        return "end" if state.get("outcome") == "invocation_ready_to_end" else "agent"

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
                invocation_count=update.get(
                    "invocation_count", previous.get("invocation_count", 0)
                ),
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

    def report_invocation_finished(self, state: AutonomousAgentState, summary: str) -> None:
        """Emit the durable state returned at the end of one graph invocation."""
        self.reporter.emit(
            InvocationFinishedEvent(
                data_source=self.source.name,
                invocation_count=state.get("invocation_count", 0),
                tool_count=state.get("tool_count", 0),
                summary=summary,
                outcome=state.get("outcome", "invocation_finished"),
                source_state=self.source_state(state),
            )
        )


def _turn(state: AutonomousAgentState, source: DataSource) -> AgentTurn:
    """Validate and reconstruct the latest dynamically typed agent turn."""
    raw_turn = state.get("turn")
    if raw_turn is None:
        raise ValueError("Tool execution requires an agent turn")
    payloads = raw_turn.get("actions")
    if payloads is None:
        payloads = [raw_turn["action"]]
    actions = [parse_core_tool(payload) or source.parse_tool(payload) for payload in payloads]
    return AgentTurn(rationale=raw_turn["rationale"], actions=actions)


def _expect_action[ActionT: SourceToolCall](
    state: AutonomousAgentState, source: DataSource, kind: type[ActionT]
) -> ActionT:
    """Return a chosen action after verifying its dynamically routed type."""
    action = _single_action(_turn(state, source))
    if not isinstance(action, kind):
        raise TypeError(f"Expected {kind.__name__}, received {type(action).__name__}")
    return action


def _single_action(turn: AgentTurn) -> SourceToolCall:
    """Return the sole action required by a serialized graph node."""
    if len(turn.actions) != 1:
        raise ValueError("This graph node requires exactly one action")
    return turn.actions[0]


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
