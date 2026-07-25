"""Build and run the autonomous LangGraph tool loop."""

import asyncio
from collections.abc import Awaitable, Callable
from time import time
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
    SelectObjective,
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
    completed_experiment_ids: list[str]
    objective: dict[str, Any] | None
    monitoring: dict[str, dict[str, Any]]
    evidence: dict[str, dict[str, Any]]
    phase: str
    objective_waited: bool
    poll_interval_seconds: int | None


Sleep = Callable[[float], Awaitable[None]]


def build_autonomous_graph(
    agent: AutonomousReasoningEngine,
    foundry: FoundryClient,
    sink: NotificationSink,
    *,
    destination: str = "local-console",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    sleep: Sleep = asyncio.sleep,
    clock: Callable[[], float] = time,
    max_wait_seconds: int | None = None,
    max_tool_calls_per_cycle: int = 32,
) -> Any:
    """Build a supervisor whose LLM selects every API, action, and lifecycle tool."""

    async def reason(state: AutonomousAgentState) -> AutonomousAgentState:
        """Let the LLM inspect prior tool results and select its next tool call."""
        turn = await agent.decide(
            state.get("transcript", []),
            state.get("completed_experiment_ids", []),
            state.get("monitoring", {}),
            state.get("evidence", {}),
        )
        print(f"[agent] {turn.action.tool}: {turn.rationale}")
        return {"turn": turn.model_dump(mode="json"), "outcome": "tool_selected"}

    def route_tool(state: AutonomousAgentState) -> str:
        """Route only tools exposed by the graph's current deterministic phase."""
        tool = _turn(state).action.tool
        phase = state.get("phase", "discovery")
        allowed = {
            "discovery": {"list_experiments", "wait", "finish_cycle"},
            "objective": {"select_objective"},
            "active": {
                "get_experiment",
                "list_experiment_updates",
                "list_experiment_results",
                "send_update",
                "wait",
                "finish_cycle",
            },
        }
        return tool if tool in allowed[phase] else "reject_action"

    async def list_experiments(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-discovery operation."""
        action = _expect_action(state, ListExperiments)
        page = await foundry.list_experiments(limit=action.limit, offset=action.offset)
        completed = set(state.get("completed_experiment_ids", []))
        monitoring = state.get("monitoring", {})
        now = clock()
        actionable = [
            item
            for item in page.items
            if item.id not in completed
            and monitoring.get(item.id, {}).get("next_poll_at", 0) <= now
        ]
        removed_count = len(page.items) - len(actionable)
        result = page.model_dump(mode="json")
        result.update(items=[item.model_dump(mode="json") for item in actionable])
        result.update(count=len(actionable), total=max(0, page.total - removed_count))
        update = _tool_result(state, action, result)
        evidence = _evidence_copy(state)
        for item in actionable:
            evidence.setdefault(item.id, {}).update(
                status=item.status.value,
                results_status=item.results_status.value,
                observed_at=now,
            )
        update.update(
            evidence=evidence,
            phase="objective" if actionable else "discovery",
        )
        return update

    async def get_experiment(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-detail operation."""
        action = _expect_action(state, GetExperiment)
        if action.experiment_id not in _objective_ids(state):
            return _rejected_tool_result(state, action, "Experiment is outside objective scope.")
        experiment = await foundry.get_experiment(action.experiment_id)
        update = _tool_result(state, action, experiment.model_dump(mode="json"))
        evidence = _evidence_copy(state)
        evidence.setdefault(action.experiment_id, {}).update(
            status=experiment.status.value,
            results_status=experiment.results_status.value,
            observed_at=clock(),
        )
        monitoring = _monitoring_copy(state)
        monitoring.setdefault(action.experiment_id, {}).update(
            last_checked_at=clock(), last_observed_status=experiment.status.value
        )
        update.update(monitoring=monitoring, evidence=evidence)
        return update

    async def select_objective(state: AutonomousAgentState) -> AutonomousAgentState:
        """Persist the validated experiment scope selected for this cycle."""
        action = _expect_action(state, SelectObjective)
        discovery = next(
            (
                entry
                for entry in reversed(state.get("transcript", []))
                if entry["tool"] == "list_experiments" and "items" in entry["result"]
            ),
            None,
        )
        discovered_ids = (
            {item["id"] for item in discovery["result"]["items"]} if discovery else set()
        )
        selected_ids = set(action.experiment_ids)
        if len(selected_ids) != len(action.experiment_ids):
            return _rejected_tool_result(
                state, action, "Objective experiment identifiers must be unique."
            )
        if not selected_ids.issubset(discovered_ids):
            return _rejected_tool_result(
                state, action, "Objective contains an experiment absent from discovery."
            )
        if action.kind == "monitor" and len(selected_ids) != 1:
            return _rejected_tool_result(
                state, action, "A monitor objective requires exactly one experiment."
            )
        if action.kind == "status_digest" and len(selected_ids) < 2:
            return _rejected_tool_result(
                state, action, "A status digest requires at least two experiments."
            )
        objective = action.model_dump(mode="json", exclude={"tool"})
        update = _tool_result(state, action, objective)
        update.update(
            objective=objective,
            phase="active",
            objective_waited=False,
            poll_interval_seconds=None,
        )
        return update

    async def list_experiment_updates(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-updates operation."""
        action = _expect_action(state, ListExperimentUpdates)
        if action.experiment_id not in _objective_ids(state):
            return _rejected_tool_result(state, action, "Experiment is outside objective scope.")
        page = await foundry.list_experiment_updates(action.experiment_id)
        update = _tool_result(state, action, page.model_dump(mode="json"))
        evidence = _evidence_copy(state)
        evidence.setdefault(action.experiment_id, {}).update(
            update_count=page.count, updates_observed_at=clock()
        )
        update["evidence"] = evidence
        return update

    async def list_experiment_results(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the documented Foundry experiment-results operation."""
        action = _expect_action(state, ListExperimentResults)
        if action.experiment_id not in _objective_ids(state):
            return _rejected_tool_result(state, action, "Experiment is outside objective scope.")
        page = await foundry.list_experiment_results(action.experiment_id)
        update = _tool_result(state, action, page.model_dump(mode="json"))
        evidence = _evidence_copy(state)
        evidence.setdefault(action.experiment_id, {}).update(
            result_count=page.count, results_observed_at=clock()
        )
        update["evidence"] = evidence
        return update

    async def send_update(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the trusted operator-update action requested by the agent."""
        action = _expect_action(state, SendUpdate)
        if not set(action.experiment_ids).issubset(_objective_ids(state)):
            return _rejected_tool_result(state, action, "Experiment is outside objective scope.")
        objective = state.get("objective") or {}
        evidence = state.get("evidence", {})
        results_ready = {
            experiment_id: evidence.get(experiment_id, {}).get("result_count", 0) > 0
            or (
                evidence.get(experiment_id, {}).get("status") == "Done"
                and evidence.get(experiment_id, {}).get("results_status") in {"Partial", "All"}
            )
            for experiment_id in action.experiment_ids
        }
        if objective.get("kind") == "report_results" and not all(results_ready.values()):
            return _rejected_tool_result(
                state, action, "Result reports require evidence for every experiment."
            )
        if (
            objective.get("kind") == "monitor"
            and not any(results_ready.values())
            and not state.get("objective_waited", False)
        ):
            return _rejected_tool_result(
                state, action, "Active monitoring requires a polling wait before reporting."
            )
        monitoring = state.get("monitoring", {})
        if objective.get("kind") == "monitor" and not any(results_ready.values()):
            unchanged_ids = [
                experiment_id
                for experiment_id in action.experiment_ids
                if monitoring.get(experiment_id, {}).get("last_reported_status")
                == evidence.get(experiment_id, {}).get("status")
            ]
            if unchanged_ids:
                return _rejected_tool_result(
                    state, action, "An unchanged monitor status was already reported."
                )
        completed = state.get("completed_experiment_ids", [])
        if set(action.experiment_ids).issubset(completed):
            update = _tool_result(
                state,
                action,
                {"status": "skipped", "reason": "experiment results already delivered"},
            )
        else:
            receipt = await sink.send(
                destination,
                Notification(title=action.title, body=action.body, priority=action.priority),
            )
            update = _tool_result(state, action, receipt.model_dump(mode="json"))
            evidenced_completions = [
                experiment_id
                for experiment_id in action.experiment_ids
                if results_ready[experiment_id]
            ]
            if evidenced_completions:
                update["completed_experiment_ids"] = list(
                    dict.fromkeys([*completed, *evidenced_completions])
                )
        update.update(
            cycle_summary=f"Delivered update for {', '.join(action.experiment_ids)}.",
            cycle_count=state.get("cycle_count", 0) + 1,
            objective=None,
            phase="discovery",
            objective_waited=False,
            poll_interval_seconds=None,
            tool_count=0,
            outcome="cycle_finished",
        )
        monitoring = _monitoring_copy(state)
        next_interval = state.get("poll_interval_seconds")
        for experiment_id in action.experiment_ids:
            record = monitoring.setdefault(experiment_id, {})
            observed_status = evidence.get(experiment_id, {}).get("status")
            if observed_status is not None:
                record["last_reported_status"] = observed_status
            if next_interval is not None:
                record["next_poll_at"] = clock() + next_interval
        update["monitoring"] = monitoring
        return update

    async def wait(state: AutonomousAgentState) -> AutonomousAgentState:
        """Execute the agent-controlled pause and return completion to the agent."""
        action = _expect_action(state, Wait)
        elapsed_seconds = min(action.seconds, max_wait_seconds or action.seconds)
        next_poll_at = clock() + elapsed_seconds
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
        monitoring = _monitoring_copy(state)
        objective = state.get("objective")
        if objective:
            for experiment_id in objective["experiment_ids"]:
                monitoring.setdefault(experiment_id, {})["next_poll_at"] = next_poll_at
        update.update(
            monitoring=monitoring,
            objective_waited=True,
            poll_interval_seconds=action.seconds,
        )
        return update

    async def finish_cycle(state: AutonomousAgentState) -> AutonomousAgentState:
        """End this invocation when the agent explicitly completes its objective."""
        action = _expect_action(state, FinishCycle)
        objective = state.get("objective")
        if (
            objective
            and objective["kind"] == "monitor"
            and state.get("tool_count", 0) < max_tool_calls_per_cycle
        ):
            return _rejected_tool_result(
                state, action, "A monitor remains active until delivery or budget yield."
            )
        return {
            "cycle_summary": action.summary,
            "cycle_count": state.get("cycle_count", 0) + 1,
            "objective": None,
            "phase": "discovery",
            "objective_waited": False,
            "poll_interval_seconds": None,
            "tool_count": 0,
            "outcome": "cycle_finished",
        }

    async def reject_action(state: AutonomousAgentState) -> AutonomousAgentState:
        """Return a deterministic policy rejection to the agent without side effects."""
        action = _turn(state).action
        phase = state.get("phase", "discovery")
        return _rejected_tool_result(
            state, action, f"Tool {action.tool} is unavailable during {phase} phase."
        )

    builder = StateGraph(AutonomousAgentState)
    builder.add_node("agent", reason)
    tools = {
        "list_experiments": list_experiments,
        "select_objective": select_objective,
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
    builder.add_node("reject_action", reject_action)
    builder.add_conditional_edges(
        "agent", route_tool, {**{name: name for name in tools}, "reject_action": "reject_action"}
    )
    for name in tools:
        if name in {"send_update", "finish_cycle"}:
            builder.add_conditional_edges(
                name,
                lambda state: "end" if state.get("outcome") == "cycle_finished" else "agent",
                {"end": END, "agent": "agent"},
            )
        else:
            builder.add_edge(name, "agent")
    builder.add_edge("reject_action", "agent")
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


def _monitoring_copy(state: AutonomousAgentState) -> dict[str, dict[str, Any]]:
    """Copy durable monitoring records before updating nested experiment state."""
    return {
        experiment_id: dict(record) for experiment_id, record in state.get("monitoring", {}).items()
    }


def _evidence_copy(state: AutonomousAgentState) -> dict[str, dict[str, Any]]:
    """Copy typed operational evidence before a tool node records observations."""
    return {
        experiment_id: dict(record) for experiment_id, record in state.get("evidence", {}).items()
    }


def _objective_ids(state: AutonomousAgentState) -> set[str]:
    """Return the experiment identifiers owned by the active graph objective."""
    objective = state.get("objective")
    return set(objective["experiment_ids"]) if objective else set()


def _rejected_tool_result(
    state: AutonomousAgentState, action: AgentToolCall, reason: str
) -> AutonomousAgentState:
    """Record a deterministic graph rejection without executing side effects."""
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
        """Run fresh cycles until cancelled or an optional demo limit is reached."""
        completed = 0
        result: dict[str, Any] = {}
        while max_cycles is None or completed < max_cycles:
            result = await self._graph.ainvoke(
                {"transcript": [], "phase": "discovery"}, config=self._config
            )
            completed += 1
        return AutonomousAgentState(**result)
