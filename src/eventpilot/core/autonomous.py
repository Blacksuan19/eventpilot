"""Build and run the generic autonomous LangGraph tool loop."""

import asyncio
from collections.abc import Awaitable, Callable
from time import time
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from eventpilot.core.agent_reasoning import (
    AgentTurn,
    AutonomousReasoningEngine,
    FinishCycle,
    SendAlert,
    Wait,
    parse_core_tool,
)
from eventpilot.core.notifications import Notification
from eventpilot.notifications.base import NotificationSink
from eventpilot.sources.base import DataSource, SourceContext, SourceToolCall


class AutonomousAgentState(TypedDict, total=False):
    """Persist generic loop state and opaque data-source state on one thread."""

    transcript: list[dict[str, Any]]
    turn: dict[str, Any]
    source_state: dict[str, Any]
    outcome: str
    cycle_summary: str | None
    cycle_count: int
    tool_count: int


Sleep = Callable[[float], Awaitable[None]]


def build_autonomous_graph(
    agent: AutonomousReasoningEngine,
    source: DataSource,
    sink: NotificationSink,
    *,
    destination: str = "local-console",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    sleep: Sleep = asyncio.sleep,
    clock: Callable[[], float] = time,
    max_wait_seconds: int | None = None,
    max_tool_calls_per_cycle: int = 32,
) -> Any:
    """Build a generic supervisor around a registered monitoring data source."""

    def source_state(state: AutonomousAgentState) -> dict[str, Any]:
        """Return persisted source state or the plugin's initial state."""
        return state.get("source_state", source.initial_state())

    async def reason(state: AutonomousAgentState) -> AutonomousAgentState:
        """Let the reasoning engine select one core or source-provided tool."""
        turn = await agent.decide(state.get("transcript", []), source_state(state))
        print(f"[agent] {turn.action.tool_name}: {turn.rationale}")
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
        return update

    async def send_alert(state: AutonomousAgentState) -> AutonomousAgentState:
        """Deliver an alert after the source validates its resource evidence."""
        action = _expect_action(state, source, SendAlert)
        current_source_state = source_state(state)
        rejection = source.validate_alert(action.resource_ids, current_source_state)
        if rejection:
            return _rejected_tool_result(state, action, rejection)
        receipt = await sink.send(
            destination,
            Notification(title=action.title, body=action.body, priority=action.priority),
        )
        update = _tool_result(state, action, receipt.model_dump(mode="json"))
        update.update(
            source_state=source.record_alert(
                action.resource_ids, current_source_state, delivered_at=clock()
            ),
            cycle_summary=f"Delivered alert for {', '.join(action.resource_ids)}.",
            cycle_count=state.get("cycle_count", 0) + 1,
            tool_count=0,
            outcome="cycle_finished",
        )
        return update

    async def wait(state: AutonomousAgentState) -> AutonomousAgentState:
        """Pause for the model-selected interval and notify the source scheduler."""
        action = _expect_action(state, source, Wait)
        elapsed_seconds = min(action.seconds, max_wait_seconds or action.seconds)
        wake_at = clock() + elapsed_seconds
        await sleep(elapsed_seconds)
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
            return _rejected_tool_result(state, action, rejection)
        return {
            "cycle_summary": action.summary,
            "cycle_count": state.get("cycle_count", 0) + 1,
            "source_state": source.record_finish(source_state(state)),
            "tool_count": 0,
            "outcome": "cycle_finished",
        }

    async def reject_action(state: AutonomousAgentState) -> AutonomousAgentState:
        """Reject an unavailable plugin tool without executing side effects."""
        action = _turn(state, source).action
        return _rejected_tool_result(
            state,
            action,
            f"Tool {action.tool_name} is unavailable in the current {source.name} state.",
        )

    builder = StateGraph(AutonomousAgentState)
    builder.add_node("agent", reason)
    builder.add_node("source_tool", source_tool)
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
            "send_alert": "send_alert",
            "wait": "wait",
            "finish_cycle": "finish_cycle",
            "reject_action": "reject_action",
        },
    )
    builder.add_edge("source_tool", "agent")
    builder.add_edge("wait", "agent")
    builder.add_edge("reject_action", "agent")
    for node in ("send_alert", "finish_cycle"):
        builder.add_conditional_edges(
            node,
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

    def __init__(self, graph: Any, *, recursion_limit: int = 10_000) -> None:
        """Bind the process loop to the global autonomous-agent thread."""
        self._graph = graph
        self._config = {
            "configurable": {"thread_id": "eventpilot-supervisor"},
            "recursion_limit": recursion_limit,
        }

    async def run(self, *, max_cycles: int | None = None) -> AutonomousAgentState:
        """Run fresh cycles until cancelled or an optional demonstration limit."""
        completed = 0
        result: dict[str, Any] = {}
        while max_cycles is None or completed < max_cycles:
            result = await self._graph.ainvoke({"transcript": []}, config=self._config)
            completed += 1
        return AutonomousAgentState(**result)
