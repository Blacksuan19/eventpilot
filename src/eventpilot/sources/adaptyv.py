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
    AcceptExperimentQuote,
    GetExperiment,
    GetExperimentQuote,
    ListExperimentResults,
    ListExperiments,
    ListExperimentUpdates,
    SubmitExperiment,
    UpdateExperiment,
)
from eventpilot.core.approvals import ApprovalRequest
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
        if phase == "active":
            statuses = {
                experiment_id: evidence.get("status")
                for experiment_id, evidence in state.get("evidence", {}).items()
                if evidence.get("detail_observed_at") is not None
            }
            if ExperimentStatus.DRAFT in statuses.values():
                tools.update({"update_experiment", "submit_experiment"})
            if ExperimentStatus.QUOTE_SENT in statuses.values():
                tools.add("get_experiment_quote")
                if any(
                    evidence.get("quote_observed_at") is not None
                    for evidence in state.get("evidence", {}).values()
                    if evidence.get("status") == ExperimentStatus.QUOTE_SENT
                ):
                    tools.add("accept_experiment_quote")
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
        if isinstance(action, UpdateExperiment):
            return await self._update_experiment(action, context)
        if isinstance(action, SubmitExperiment):
            return await self._submit_experiment(action, context)
        if isinstance(action, AcceptExperimentQuote):
            return await self._accept_experiment_quote(action, context)
        if isinstance(action, GetExperimentQuote):
            return await self._get_experiment_quote(action, context)
        raise TypeError(f"Unsupported {self.name} action: {type(action).__name__}")

    def approval_request(
        self, action: SourceToolCall, state: dict[str, Any]
    ) -> ApprovalRequest | None:
        """Require operator confirmation before accepting a quote and creating an invoice."""
        if not isinstance(action, AcceptExperimentQuote):
            return None
        evidence = state.get("evidence", {}).get(action.experiment_id, {})
        quote = evidence.get("quote", {})
        amount = quote.get("amount_total")
        currency = str(quote.get("currency", "")).upper()
        formatted_amount = (
            f"{currency} {amount / 100:,.2f}" if isinstance(amount, int) else "unknown amount"
        )
        return ApprovalRequest(
            title=f"Approve quote for {action.experiment_id}",
            body=(
                f"Foundry quoted {formatted_amount} for {action.experiment_id}. "
                "Approval will accept the quote and create an invoice."
            ),
            resource_ids=(action.experiment_id,),
        )

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
            detail_observed_at=now,
            experiment_spec=result.get("experiment_spec", {}),
        )
        state.setdefault("monitoring", {}).setdefault(action.experiment_id, {}).update(
            last_checked_at=now,
            last_observed_status=result["status"],
        )
        self._record_inspection(state, action.experiment_id)
        return SourceExecution(result=result, state=state)

    async def _update_experiment(
        self, action: UpdateExperiment, context: SourceContext
    ) -> SourceExecution:
        """Modify an editable in-scope experiment and persist the returned configuration."""
        rejection = self._scope_rejection(action.experiment_id, context.state)
        evidence = context.state.get("evidence", {}).get(action.experiment_id, {})
        if rejection:
            return SourceExecution(result=rejection, state=context.state)
        if evidence.get("status") not in {
            ExperimentStatus.DRAFT,
            ExperimentStatus.IN_REVIEW,
        }:
            return SourceExecution(
                result={"status": "rejected", "reason": "Experiment is not editable."},
                state=context.state,
            )
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        state.setdefault("evidence", {}).setdefault(action.experiment_id, {}).update(
            status=result["status"],
            experiment_spec=result.get("experiment_spec", {}),
            detail_observed_at=context.clock(),
            last_action="update_experiment",
        )
        self._record_inspection(state, action.experiment_id)
        return SourceExecution(result=result, state=state)

    async def _submit_experiment(
        self, action: SubmitExperiment, context: SourceContext
    ) -> SourceExecution:
        """Submit a valid in-scope draft and persist its confirmed status transition."""
        rejection = self._scope_rejection(action.experiment_id, context.state)
        evidence = context.state.get("evidence", {}).get(action.experiment_id, {})
        if rejection:
            return SourceExecution(result=rejection, state=context.state)
        if evidence.get("status") != ExperimentStatus.DRAFT:
            return SourceExecution(
                result={"status": "rejected", "reason": "Experiment is not a draft."},
                state=context.state,
            )
        replicates = evidence.get("experiment_spec", {}).get("n_replicates", 0)
        if replicates < 2:
            return SourceExecution(
                result={
                    "status": "rejected",
                    "reason": "Draft policy requires at least two replicates before submission.",
                },
                state=context.state,
            )
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        state.setdefault("evidence", {}).setdefault(action.experiment_id, {}).update(
            status=result["status"],
            observed_at=context.clock(),
            last_action="submit_experiment",
        )
        self._record_inspection(state, action.experiment_id)
        return SourceExecution(result=result, state=state)

    async def _accept_experiment_quote(
        self, action: AcceptExperimentQuote, context: SourceContext
    ) -> SourceExecution:
        """Accept an approved in-scope quote and persist its invoice evidence."""
        rejection = self._scope_rejection(action.experiment_id, context.state)
        evidence = context.state.get("evidence", {}).get(action.experiment_id, {})
        if rejection:
            return SourceExecution(result=rejection, state=context.state)
        if evidence.get("status") != ExperimentStatus.QUOTE_SENT:
            return SourceExecution(
                result={"status": "rejected", "reason": "Experiment has no pending quote."},
                state=context.state,
            )
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        state.setdefault("evidence", {}).setdefault(action.experiment_id, {}).update(
            status=ExperimentStatus.WAITING_FOR_MATERIALS,
            observed_at=context.clock(),
            last_action="accept_experiment_quote",
            invoice_id=result.get("invoice_id"),
        )
        self._record_inspection(state, action.experiment_id)
        return SourceExecution(result=result, state=state)

    async def _get_experiment_quote(
        self, action: GetExperimentQuote, context: SourceContext
    ) -> SourceExecution:
        """Read quote cost and expiry before exposing its consequential acceptance tool."""
        rejection = self._scope_rejection(action.experiment_id, context.state)
        evidence = context.state.get("evidence", {}).get(action.experiment_id, {})
        if rejection:
            return SourceExecution(result=rejection, state=context.state)
        if evidence.get("status") != ExperimentStatus.QUOTE_SENT:
            return SourceExecution(
                result={"status": "rejected", "reason": "Experiment has no pending quote."},
                state=context.state,
            )
        result = await self._adapter.execute(action)
        state = deepcopy(context.state)
        state.setdefault("evidence", {}).setdefault(action.experiment_id, {}).update(
            quote=result,
            quote_observed_at=context.clock(),
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
        for experiment_id, evidence in state.get("evidence", {}).items():
            if evidence.get("detail_observed_at") is None:
                continue
            if evidence.get("status") == ExperimentStatus.DRAFT:
                return f"Draft {experiment_id} must be prepared and submitted before waiting."
            if evidence.get("status") == ExperimentStatus.QUOTE_SENT:
                return f"Quote {experiment_id} must be reviewed for approval before waiting."
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
