"""Expose Adaptyv Foundry as a pluggable autonomous monitoring source."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import Field

from eventpilot.adapters.adaptyv import ExperimentStatus, FoundryClient, FoundryToolAdapter
from eventpilot.adapters.adaptyv.tools import (
    GetExperiment,
    ListExperimentResults,
    ListExperiments,
    ListExperimentUpdates,
)
from eventpilot.core.agent_reasoning import AgentTurn, SendAlert, Wait
from eventpilot.prompts.loader import load_prompt
from eventpilot.sources.base import SourceContext, SourceExecution, SourceToolCall

ObjectiveKind = Literal["monitor", "report_results", "status_digest", "investigate_incident"]


class SelectObjective(SourceToolCall):
    """Commit the cycle to a validated experiment scope and objective type."""

    tool: Literal["select_objective"] = "select_objective"
    kind: ObjectiveKind = Field(description="Monitoring objective enforced for this cycle.")
    experiment_ids: list[str] = Field(
        min_length=1, description="Discovered experiment identifiers included in the objective."
    )
    summary: str = Field(min_length=1, description="Purpose and scope of this objective.")


class AdaptyvDataSource:
    """Implement Foundry tools and deterministic experiment-monitoring policy."""

    name = "adaptyv-foundry"

    def __init__(self, client: FoundryClient) -> None:
        """Bind the source plugin to a real or fixture-backed Foundry client."""
        self._adapter = FoundryToolAdapter(client)
        self.tool_types: tuple[type[SourceToolCall], ...] = (
            *self._adapter.tool_types,
            SelectObjective,
        )
        self.instructions = load_prompt("sources/adaptyv.txt")
        self._tool_types = {
            tool_type.model_fields["tool"].default: tool_type for tool_type in self.tool_types
        }

    def initial_state(self) -> dict[str, Any]:
        """Create empty durable monitoring state for a Foundry organization."""
        return {
            "phase": "discovery",
            "completed_experiment_ids": [],
            "monitoring": {},
            "evidence": {},
            "objective": None,
            "objective_waited": False,
            "poll_interval_seconds": None,
        }

    def available_tools(self, state: dict[str, Any]) -> set[str]:
        """Expose only Foundry operations valid in the current objective phase."""
        phase = state.get("phase", "discovery")
        return {
            "discovery": {"list_experiments"},
            "objective": {"select_objective"},
            "active": {
                "get_experiment",
                "list_experiment_updates",
                "list_experiment_results",
            },
        }[phase]

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate a persisted Foundry tool call by its discriminator."""
        tool_type = self._tool_types.get(payload.get("tool"))
        if tool_type is None:
            raise ValueError(f"Unknown {self.name} tool: {payload.get('tool')}")
        return tool_type.model_validate(payload)

    async def execute(self, action: SourceToolCall, context: SourceContext) -> SourceExecution:
        """Dispatch one validated Foundry tool to its source-owned handler."""
        if isinstance(action, ListExperiments):
            return await self._list_experiments(action, context)
        if isinstance(action, SelectObjective):
            return self._select_objective(action, context)
        if isinstance(action, GetExperiment):
            return await self._get_experiment(action, context)
        if isinstance(action, ListExperimentUpdates):
            return await self._list_experiment_updates(action, context)
        if isinstance(action, ListExperimentResults):
            return await self._list_experiment_results(action, context)
        raise TypeError(f"Unsupported {self.name} action: {type(action).__name__}")

    async def _list_experiments(
        self, action: ListExperiments, context: SourceContext
    ) -> SourceExecution:
        """Discover actionable experiments while respecting durable poll windows."""
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        completed = set(state.get("completed_experiment_ids", []))
        monitoring = state.get("monitoring", {})
        now = context.clock()
        actionable = [
            item
            for item in result["items"]
            if item["id"] not in completed
            and monitoring.get(item["id"], {}).get("next_poll_at", 0) <= now
        ]
        original_count = len(result["items"])
        result.update(items=actionable)
        result.update(
            count=len(actionable),
            total=max(0, result["total"] - (original_count - len(actionable))),
        )
        evidence = state.setdefault("evidence", {})
        for item in actionable:
            evidence.setdefault(item["id"], {}).update(
                status=item["status"],
                results_status=item["results_status"],
                observed_at=now,
            )
        state["phase"] = "objective" if actionable else "discovery"
        return SourceExecution(result=result, state=state)

    def _select_objective(self, action: SelectObjective, context: SourceContext) -> SourceExecution:
        """Validate and persist a scope selected from the latest discovery result."""
        state = deepcopy(context.state)
        discovery = next(
            (
                entry
                for entry in reversed(context.transcript)
                if entry["tool"] == "list_experiments" and "items" in entry["result"]
            ),
            None,
        )
        discovered_ids = (
            {item["id"] for item in discovery["result"]["items"]} if discovery else set()
        )
        selected_ids = set(action.experiment_ids)
        rejection = None
        if len(selected_ids) != len(action.experiment_ids):
            rejection = "Objective experiment identifiers must be unique."
        elif not selected_ids.issubset(discovered_ids):
            rejection = "Objective contains an experiment absent from discovery."
        elif action.kind == "monitor" and len(selected_ids) != 1:
            rejection = "A monitor objective requires exactly one experiment."
        elif action.kind == "status_digest" and len(selected_ids) < 2:
            rejection = "A status digest requires at least two experiments."
        if rejection:
            return SourceExecution(result={"status": "rejected", "reason": rejection}, state=state)
        objective = action.model_dump(mode="json", exclude={"tool"})
        state.update(
            objective=objective,
            phase="active",
            objective_waited=False,
            poll_interval_seconds=None,
        )
        return SourceExecution(result=objective, state=state)

    async def _get_experiment(
        self, action: GetExperiment, context: SourceContext
    ) -> SourceExecution:
        """Fetch experiment detail and persist current status evidence."""
        rejection = self._scope_rejection(action.experiment_id, context.state)
        if rejection:
            return SourceExecution(result=rejection, state=context.state)
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        now = context.clock()
        state.setdefault("evidence", {}).setdefault(action.experiment_id, {}).update(
            status=result["status"],
            results_status=result["results_status"],
            observed_at=now,
        )
        state.setdefault("monitoring", {}).setdefault(action.experiment_id, {}).update(
            last_checked_at=now,
            last_observed_status=result["status"],
        )
        return SourceExecution(result=result, state=state)

    async def _list_experiment_updates(
        self, action: ListExperimentUpdates, context: SourceContext
    ) -> SourceExecution:
        """Fetch updates and persist evidence that the endpoint was inspected."""
        rejection = self._scope_rejection(action.experiment_id, context.state)
        if rejection:
            return SourceExecution(result=rejection, state=context.state)
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        state.setdefault("evidence", {}).setdefault(action.experiment_id, {}).update(
            update_count=result["count"],
            updates_observed_at=context.clock(),
        )
        return SourceExecution(result=result, state=state)

    async def _list_experiment_results(
        self, action: ListExperimentResults, context: SourceContext
    ) -> SourceExecution:
        """Fetch results and persist evidence that operator-ready data was inspected."""
        rejection = self._scope_rejection(action.experiment_id, context.state)
        if rejection:
            return SourceExecution(result=rejection, state=context.state)
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        state.setdefault("evidence", {}).setdefault(action.experiment_id, {}).update(
            result_count=result["count"],
            results_observed_at=context.clock(),
        )
        return SourceExecution(result=result, state=state)

    def after_wait(
        self, state: dict[str, Any], *, requested_seconds: int, wake_at: float
    ) -> dict[str, Any]:
        """Schedule the active experiments for their next model-selected poll."""
        updated = deepcopy(state)
        objective = updated.get("objective")
        if objective:
            monitoring = updated.setdefault("monitoring", {})
            for experiment_id in objective["experiment_ids"]:
                monitoring.setdefault(experiment_id, {})["next_poll_at"] = wake_at
        updated["objective_waited"] = True
        updated["poll_interval_seconds"] = requested_seconds
        return updated

    def validate_alert(self, resource_ids: list[str], state: dict[str, Any]) -> str | None:
        """Require scoped, current Foundry evidence before an operator alert."""
        objective = state.get("objective") or {}
        if not set(resource_ids).issubset(set(objective.get("experiment_ids", []))):
            return "Experiment is outside objective scope."
        evidence = state.get("evidence", {})
        results_ready = self._results_ready(resource_ids, evidence)
        if objective.get("kind") == "report_results" and not all(results_ready.values()):
            return "Result reports require evidence for every experiment."
        if (
            objective.get("kind") == "monitor"
            and not any(results_ready.values())
            and not state.get("objective_waited", False)
        ):
            return "Active monitoring requires a polling wait before reporting."
        monitoring = state.get("monitoring", {})
        if objective.get("kind") == "monitor" and not any(results_ready.values()):
            unchanged = [
                resource_id
                for resource_id in resource_ids
                if monitoring.get(resource_id, {}).get("last_reported_status")
                == evidence.get(resource_id, {}).get("status")
            ]
            if unchanged:
                return "An unchanged monitor status was already reported."
        return None

    def record_alert(
        self, resource_ids: list[str], state: dict[str, Any], *, delivered_at: float
    ) -> dict[str, Any]:
        """Record delivered results and schedule later monitoring after an alert."""
        updated = deepcopy(state)
        evidence = updated.get("evidence", {})
        ready = self._results_ready(resource_ids, evidence)
        completed = updated.setdefault("completed_experiment_ids", [])
        updated["completed_experiment_ids"] = list(
            dict.fromkeys([*completed, *[item for item in resource_ids if ready[item]]])
        )
        monitoring = updated.setdefault("monitoring", {})
        next_interval = updated.get("poll_interval_seconds")
        for resource_id in resource_ids:
            record = monitoring.setdefault(resource_id, {})
            observed_status = evidence.get(resource_id, {}).get("status")
            if observed_status is not None:
                record["last_reported_status"] = observed_status
            if next_interval is not None:
                record["next_poll_at"] = delivered_at + next_interval
        updated.update(
            objective=None,
            phase="discovery",
            objective_waited=False,
            poll_interval_seconds=None,
        )
        return updated

    def validate_finish(
        self, state: dict[str, Any], *, tool_count: int, max_tool_calls: int
    ) -> str | None:
        """Keep longitudinal monitoring active until delivery or a budget yield."""
        objective = state.get("objective")
        if objective and objective["kind"] == "monitor" and tool_count < max_tool_calls:
            return "A monitor remains active until delivery or budget yield."
        return None

    def record_finish(self, state: dict[str, Any]) -> dict[str, Any]:
        """Clear bounded objective state before the next Foundry discovery cycle."""
        updated = deepcopy(state)
        updated.update(
            objective=None,
            phase="discovery",
            objective_waited=False,
            poll_interval_seconds=None,
        )
        return updated

    @staticmethod
    def _scope_rejection(experiment_id: str, state: dict[str, Any]) -> dict[str, Any] | None:
        """Reject an experiment operation outside the active objective scope."""
        objective = state.get("objective")
        scoped_ids = set(objective["experiment_ids"]) if objective else set()
        if experiment_id not in scoped_ids:
            return {"status": "rejected", "reason": "Experiment is outside objective scope."}
        return None

    @staticmethod
    def _results_ready(
        resource_ids: list[str], evidence: dict[str, dict[str, Any]]
    ) -> dict[str, bool]:
        """Return result-readiness derived from explicit Foundry observations."""
        return {
            resource_id: evidence.get(resource_id, {}).get("result_count", 0) > 0
            or (
                evidence.get(resource_id, {}).get("status") == "Done"
                and evidence.get(resource_id, {}).get("results_status") in {"Partial", "All"}
            )
            for resource_id in resource_ids
        }


class DemoAdaptyvReasoningEngine:
    """Exercise the Adaptyv plugin deterministically without LLM credentials."""

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Choose a representative Foundry trajectory using only observed tool results."""
        if not transcript:
            return self._discovery_turn()

        latest = transcript[-1]
        tool = latest["tool"]
        if tool == "list_experiments":
            items = latest["result"]["items"]
            active = [
                item
                for item in items
                if item["id"] not in source_state.get("completed_experiment_ids", [])
                if item["status"] != ExperimentStatus.CANCELED
            ]
            if not active:
                return AgentTurn(
                    rationale="No active experiment currently requires attention.",
                    action=Wait(seconds=60, reason="Back off before discovering work again."),
                )
            experiment_id = next(
                (item["id"] for item in active if item["status"] != ExperimentStatus.DONE),
                active[0]["id"],
            )
            return AgentTurn(
                rationale="Commit this cycle to one discovered experiment.",
                action=SelectObjective(
                    kind="monitor",
                    experiment_ids=[experiment_id],
                    summary=f"Monitor experiment {experiment_id} until it is actionable.",
                ),
            )

        if tool == "select_objective":
            experiment_id = latest["result"]["experiment_ids"][0]
            return AgentTurn(
                rationale="Inspect the experiment selected for this cycle.",
                action=GetExperiment(experiment_id=experiment_id),
            )

        if tool == "wait":
            experiment_id = self._last_focused_experiment_id(transcript)
            if experiment_id:
                return AgentTurn(
                    rationale="The polling interval elapsed; refresh the active experiment.",
                    action=GetExperiment(experiment_id=experiment_id),
                )
            return self._discovery_turn()

        if tool == "get_experiment":
            experiment = latest["result"]
            experiment_id = experiment["id"]
            if experiment["results_status"] in {"Partial", "All"}:
                return AgentTurn(
                    rationale="Results are available and must be inspected before alerting.",
                    action=ListExperimentResults(experiment_id=experiment_id),
                )
            if experiment["status"] == ExperimentStatus.DONE:
                return AgentTurn(
                    rationale="The experiment is done but results are delayed; inspect updates.",
                    action=ListExperimentUpdates(experiment_id=experiment_id),
                )
            return AgentTurn(
                rationale=f"The experiment remains active in {experiment['status']}.",
                action=Wait(seconds=1, reason="Poll this active experiment again."),
            )

        if tool == "list_experiment_updates":
            experiment_id = latest["call"]["experiment_id"]
            return AgentTurn(
                rationale="The terminal experiment has no results yet and needs another poll.",
                action=Wait(seconds=5, reason=f"Wait for results on {experiment_id}."),
            )

        if tool == "list_experiment_results":
            experiment_id = latest["call"]["experiment_id"]
            return AgentTurn(
                rationale="The inspected result payload is ready for an operator alert.",
                action=SendAlert(
                    resource_ids=[experiment_id],
                    title=f"Experiment {experiment_id} results are ready",
                    body=f"Foundry returned {latest['result']['count']} available result(s).",
                ),
            )

        return self._discovery_turn()

    @staticmethod
    def _discovery_turn() -> AgentTurn:
        """Return the first Foundry discovery action for a fresh cycle."""
        return AgentTurn(
            rationale="Discover current experiments before selecting an objective.",
            action=ListExperiments(),
        )

    @staticmethod
    def _last_focused_experiment_id(transcript: list[dict[str, Any]]) -> str | None:
        """Return the experiment most recently inspected in this cycle."""
        for entry in reversed(transcript):
            if entry["tool"] == "get_experiment":
                return str(entry["result"]["id"])
        return None
