"""Build and run the autonomous LangGraph tool loop."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from eventpilot.adapters.adaptyv import FoundryClient
from eventpilot.core.agent_reasoning import (
    AgentToolCall,
    AgentTurn,
    AutonomousReasoningEngine,
    FinishCycle,
    GetExperiment,
    ListExperimentResults,
    ListExperiments,
    ListExperimentUpdates,
    SendUpdate,
    Wait,
)
from eventpilot.core.notifications import Notification
from eventpilot.notifications.base import NotificationSink


class AutonomousAgentState(TypedDict, total=False):
    """Persist tool context and cycle outcomes on one supervisor thread."""

    transcript: list[dict[str, Any]]
    turn: dict[str, Any]
    outcome: str
    cycle_summary: str | None
    cycle_count: int
    tool_count: int


Sleep = Callable[[float], Awaitable[None]]


def build_autonomous_graph(
    agent: AutonomousReasoningEngine,
    foundry: FoundryClient,
    sink: NotificationSink,
    *,
    destination: str = "local-console",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    sleep: Sleep = asyncio.sleep,
    max_wait_seconds: int | None = None,
) -> Any:
    """Build a supervisor whose LLM selects every API, action, and lifecycle tool."""

    async def reason(state: AutonomousAgentState) -> AutonomousAgentState:
        """Let the LLM inspect prior tool results and select its next tool call."""
        turn = await agent.decide(state.get("transcript", []))
        print(f"[agent] {turn.action.tool}: {turn.rationale}")
        return {"turn": turn.model_dump(mode="json"), "outcome": "tool_selected"}

    def route_tool(state: AutonomousAgentState) -> str:
        """Route the exact validated tool name chosen by the LLM."""
        return _turn(state).action.tool

    async def list_experiments(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-discovery operation."""
        action = _expect_action(state, ListExperiments)
        page = await foundry.list_experiments(limit=action.limit, offset=action.offset)
        return _tool_result(state, action, page.model_dump(mode="json"))

    async def get_experiment(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-detail operation."""
        action = _expect_action(state, GetExperiment)
        experiment = await foundry.get_experiment(action.experiment_id)
        return _tool_result(state, action, experiment.model_dump(mode="json"))

    async def list_experiment_updates(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-updates operation."""
        action = _expect_action(state, ListExperimentUpdates)
        page = await foundry.list_experiment_updates(action.experiment_id)
        return _tool_result(state, action, page.model_dump(mode="json"))

    async def list_experiment_results(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-results operation."""
        action = _expect_action(state, ListExperimentResults)
        page = await foundry.list_experiment_results(action.experiment_id)
        return _tool_result(state, action, page.model_dump(mode="json"))

    async def send_update(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the trusted operator-update action requested by the agent."""
        action = _expect_action(state, SendUpdate)
        receipt = await sink.send(
            destination,
            Notification(title=action.title, body=action.body, priority=action.priority),
        )
        return _tool_result(state, action, receipt.model_dump(mode="json"))

    async def wait(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the agent-controlled pause and return completion to the agent."""
        action = _expect_action(state, Wait)
        elapsed_seconds = min(action.seconds, max_wait_seconds or action.seconds)
        await sleep(elapsed_seconds)
        return _tool_result(
            state,
            action,
            {
                "status": "completed",
                "requested_seconds": action.seconds,
                "elapsed_seconds": elapsed_seconds,
                "reason": action.reason,
            },
        )

    async def finish_cycle(state: AutonomousAgentState) -> AutonomousAgentState:
        """End this invocation when the agent explicitly completes its objective."""
        action = _expect_action(state, FinishCycle)
        return {
            "cycle_summary": action.summary,
            "cycle_count": state.get("cycle_count", 0) + 1,
            "outcome": "cycle_finished",
        }

    builder = StateGraph(AutonomousAgentState)
    builder.add_node("agent", reason)
    tools = {
        "list_experiments": list_experiments,
        "get_experiment": get_experiment,
        "list_experiment_updates": list_experiment_updates,
        "list_experiment_results": list_experiment_results,
        "send_update": send_update,
        "wait": wait,
        "finish_cycle": finish_cycle,
    }
    for name, tool in tools.items():
        builder.add_node(name, tool)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_tool, {name: name for name in tools})
    for name in tools:
        builder.add_edge(name, END if name == "finish_cycle" else "agent")
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def _turn(state: AutonomousAgentState) -> AgentTurn:
    """Validate and return the latest structured agent turn from graph state."""
    raw_turn = state.get("turn")
    if raw_turn is None:
        raise ValueError("Tool execution requires an agent turn")
    return AgentTurn.model_validate(raw_turn)


def _expect_action[ActionT: AgentToolCall](
    state: AutonomousAgentState, kind: type[ActionT]
) -> ActionT:
    """Return the chosen action after verifying the routed tool type."""
    action = _turn(state).action
    if not isinstance(action, kind):
        raise TypeError(f"Expected {kind.__name__}, received {type(action).__name__}")
    return action


def _tool_result(
    state: AutonomousAgentState, action: AgentToolCall, result: dict[str, Any]
) -> AutonomousAgentState:
    """Append an executed tool call and result to the agent's working context."""
    transcript = [
        *state.get("transcript", []),
        {"tool": action.tool, "call": action.model_dump(mode="json"), "result": result},
    ]
    return {
        "transcript": transcript,
        "tool_count": state.get("tool_count", 0) + 1,
        "outcome": f"{action.tool}_completed",
    }


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
        """Run fresh cycles until cancelled or an optional demo limit is reached."""
        completed = 0
        result: dict[str, Any] = {}
        while max_cycles is None or completed < max_cycles:
            result = await self._graph.ainvoke({"transcript": []}, config=self._config)
            completed += 1
        return AutonomousAgentState(**result)
