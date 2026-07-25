"""Expose Adaptyv Foundry as a pluggable autonomous monitoring source."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import Field

from eventpilot.adapters.adaptyv import (
    ExperimentStatus,
    FoundryClient,
    FoundryToolAdapter,
)
from eventpilot.adapters.adaptyv.tools import (
    GetExperiment,
    ListExperimentResults,
    ListExperiments,
    ListExperimentUpdates,
)
from eventpilot.core.agent_reasoning import AgentTurn, FinishCycle, SendAlert, Wait
from eventpilot.sources.base import SourceContext, SourceExecution, SourceToolCall

ObjectiveKind = Literal["monitor", "report_results", "status_digest", "investigate_incident"]


class SelectObjective(SourceToolCall):
    """Commit the cycle to a validated experiment portfolio and objective type."""

    tool: Literal["select_objective"] = "select_objective"
    kind: ObjectiveKind = Field(description="Monitoring objective enforced for this cycle.")
    experiment_ids: list[str] = Field(
        min_length=1,
        description="Discovered experiment identifiers available for interleaved work.",
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
        self.instructions = self._adapter.instructions
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
            "last_inspected_experiment_id": None,
            "next_experiment_candidates": [],
            "pending_result_alert_id": None,
        }

    def available_tools(self, state: dict[str, Any]) -> set[str]:
        """Expose only Foundry operations valid in the current objective phase."""
        if state.get("pending_result_alert_id"):
            return set()
        phase = state.get("phase", "discovery")
        tools = {
            "discovery": {"list_experiments"},
            "idle": set(),
            "objective": {"select_objective"},
            "active": {
                "get_experiment",
                "list_experiment_updates",
                "list_experiment_results",
            },
        }[phase]
        if phase == "active" and self._objective_omits_discovered_work(state):
            tools.add("select_objective")
        return tools

    def is_idle(self, state: dict[str, Any]) -> bool:
        """Return whether discovery found no currently actionable experiments."""
        return state.get("phase") == "idle"

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
        state["phase"] = "objective" if actionable else "idle"
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
        elif action.kind == "monitor":
            monitorable_ids = {
                item["id"]
                for item in (discovery["result"]["items"] if discovery else [])
                if item["status"] != ExperimentStatus.CANCELED
            }
            if selected_ids != monitorable_ids:
                rejection = (
                    "A monitor objective must include every active discovered experiment so "
                    "monitoring can be interleaved."
                )
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
            last_inspected_experiment_id=None,
            next_experiment_candidates=[],
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
        self._record_inspection(state, action.experiment_id)
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
        self._record_inspection(state, action.experiment_id)
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
        self._record_inspection(state, action.experiment_id)
        if result["count"] > 0:
            state["pending_result_alert_id"] = action.experiment_id
        return SourceExecution(result=result, state=state)

    def after_wait(
        self, state: dict[str, Any], *, requested_seconds: int, wake_at: float
    ) -> dict[str, Any]:
        """Schedule the active experiments for their next model-selected poll."""
        updated = deepcopy(state)
        was_idle = updated.get("phase") == "idle"
        objective = updated.get("objective") or {}
        last_inspected = updated.get("last_inspected_experiment_id")
        if last_inspected:
            monitoring = updated.setdefault("monitoring", {})
            monitoring.setdefault(last_inspected, {})["next_poll_at"] = wake_at
        completed = set(updated.get("completed_experiment_ids", []))
        updated["next_experiment_candidates"] = [
            experiment_id
            for experiment_id in objective.get("experiment_ids", [])
            if experiment_id != last_inspected and experiment_id not in completed
        ]
        updated["objective_waited"] = True
        updated["poll_interval_seconds"] = requested_seconds
        if was_idle:
            updated["phase"] = "discovery"
        return updated

    def validate_wait(self, state: dict[str, Any]) -> str | None:
        """Prevent delaying an operator-ready result after it has been inspected."""
        pending = state.get("pending_result_alert_id")
        if pending:
            return f"Result evidence for {pending} must be reported before waiting."
        return None

    def validate_alert(self, resource_ids: list[str], state: dict[str, Any]) -> str | None:
        """Require scoped, current Foundry evidence before an operator alert."""
        pending = state.get("pending_result_alert_id")
        if pending and resource_ids != [pending]:
            return f"Report the pending result for {pending} in its own alert."
        objective = state.get("objective") or {}
        if not set(resource_ids).issubset(set(objective.get("experiment_ids", []))):
            return "Experiment is outside objective scope."
        evidence = state.get("evidence", {})
        results_ready = self._results_ready(resource_ids, evidence)
        if len(resource_ids) > 1 and any(results_ready.values()):
            return "Report each experiment's results in a separate alert."
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
        """Record delivery while preserving the current portfolio objective."""
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
        updated["pending_result_alert_id"] = None
        if not self.should_continue_after_alert(updated):
            updated.update(
                objective=None,
                phase="discovery",
                objective_waited=False,
                poll_interval_seconds=None,
                last_inspected_experiment_id=None,
                next_experiment_candidates=[],
            )
        return updated

    def should_continue_after_alert(self, state: dict[str, Any]) -> bool:
        """Continue when another objective experiment already has ready results."""
        objective = state.get("objective") or {}
        completed = set(state.get("completed_experiment_ids", []))
        evidence = state.get("evidence", {})
        return any(
            experiment_id not in completed
            and evidence.get(experiment_id, {}).get("status") == ExperimentStatus.DONE
            and evidence.get(experiment_id, {}).get("results_status") in {"Partial", "All"}
            for experiment_id in objective.get("experiment_ids", [])
        )

    def validate_finish(
        self, state: dict[str, Any], *, tool_count: int, max_tool_calls: int
    ) -> str | None:
        """Keep longitudinal monitoring active until delivery or a budget yield."""
        pending = state.get("pending_result_alert_id")
        if pending:
            return f"Result evidence for {pending} must be reported before finishing the cycle."
        if state.get("phase") == "idle":
            return "An idle source must wait before starting another discovery cycle."
        return None

    def record_finish(self, state: dict[str, Any]) -> dict[str, Any]:
        """Clear bounded objective state before the next Foundry discovery cycle."""
        updated = deepcopy(state)
        updated.update(
            objective=None,
            phase="discovery",
            objective_waited=False,
            poll_interval_seconds=None,
            last_inspected_experiment_id=None,
            next_experiment_candidates=[],
            pending_result_alert_id=None,
        )
        return updated

    @staticmethod
    def _scope_rejection(experiment_id: str, state: dict[str, Any]) -> dict[str, Any] | None:
        """Reject an experiment operation outside the active objective scope."""
        objective = state.get("objective")
        scoped_ids = set(objective["experiment_ids"]) if objective else set()
        if experiment_id not in scoped_ids:
            return {
                "status": "rejected",
                "reason": "Experiment is outside objective scope.",
            }
        candidates = state.get("next_experiment_candidates", [])
        if candidates and experiment_id == state.get("last_inspected_experiment_id"):
            return {
                "status": "rejected",
                "reason": (
                    "Inspect another portfolio experiment before returning to this one. "
                    f"Available candidates: {', '.join(candidates)}."
                ),
            }
        return None

    @staticmethod
    def _record_inspection(state: dict[str, Any], experiment_id: str) -> None:
        """Record portfolio progress and clear a rotation gate after switching experiments."""
        if experiment_id != state.get("last_inspected_experiment_id"):
            state["next_experiment_candidates"] = []
        state["last_inspected_experiment_id"] = experiment_id

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

    @staticmethod
    def _objective_omits_discovered_work(state: dict[str, Any]) -> bool:
        """Detect legacy or incomplete monitor scopes that need portfolio expansion."""
        objective = state.get("objective") or {}
        if objective.get("kind") != "monitor":
            return False
        completed = set(state.get("completed_experiment_ids", []))
        discovered = {
            experiment_id
            for experiment_id, evidence in state.get("evidence", {}).items()
            if experiment_id not in completed
            and evidence.get("status") != ExperimentStatus.CANCELED
        }
        return not discovered.issubset(set(objective.get("experiment_ids", [])))


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
            return AgentTurn(
                rationale=(
                    "Monitor the discovered experiment portfolio without blocking on one run."
                ),
                action=SelectObjective(
                    kind="monitor",
                    experiment_ids=[item["id"] for item in active],
                    summary="Interleave monitoring across all active experiments.",
                ),
            )

        if tool == "select_objective":
            experiment_id = latest["result"]["experiment_ids"][0]
            return AgentTurn(
                rationale="Inspect the experiment selected for this cycle.",
                action=GetExperiment(experiment_id=experiment_id),
            )

        if tool == "wait":
            candidates = source_state.get("next_experiment_candidates", [])
            experiment_id = (
                candidates[0] if candidates else self._last_focused_experiment_id(transcript)
            )
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

        if tool == "send_alert":
            completed = set(source_state.get("completed_experiment_ids", []))
            evidence = source_state.get("evidence", {})
            objective = source_state.get("objective") or {}
            next_ready = next(
                (
                    experiment_id
                    for experiment_id in objective.get("experiment_ids", [])
                    if experiment_id not in completed
                    and evidence.get(experiment_id, {}).get("status") == ExperimentStatus.DONE
                    and evidence.get(experiment_id, {}).get("results_status") in {"Partial", "All"}
                ),
                None,
            )
            if next_ready:
                return AgentTurn(
                    rationale="Another experiment has ready results in the current portfolio.",
                    action=ListExperimentResults(experiment_id=next_ready),
                )
            return AgentTurn(
                rationale="No other portfolio result is immediately reportable.",
                action=FinishCycle(summary="Reported all currently ready results."),
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
