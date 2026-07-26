"""Provide deterministic reasoning for the credential-free Adaptyv demo."""

from typing import Any

from eventpilot.adapters.adaptyv import ExperimentStatus
from eventpilot.adapters.adaptyv.models import ModifyExperimentRequest
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
from eventpilot.core.agent_reasoning import AgentTurn, SendAlert, Wait
from eventpilot.core.monitoring import (
    SelectObjective,
    pending_alert_resource_ids,
    required_source_actions,
)


class DemoAdaptyvReasoningEngine:
    """Exercise the Adaptyv plugin deterministically without LLM credentials."""

    async def decide(
        self, transcript: list[dict[str, Any]], source_state: dict[str, Any]
    ) -> AgentTurn:
        """Choose a representative Foundry trajectory using only observed tool results."""
        pending_alerts = pending_alert_resource_ids(source_state)
        if pending_alerts:
            experiment_id = pending_alerts[0]
            result_count = (
                source_state.get("evidence", {}).get(experiment_id, {}).get("result_count", 0)
            )
            return AgentTurn(
                rationale="Deliver the next inspected result waiting in the alert queue.",
                action=SendAlert(
                    resource_ids=[experiment_id],
                    title=f"Experiment {experiment_id} results are ready",
                    body=f"Foundry returned {result_count} available result(s).",
                ),
            )

        required = required_source_actions(source_state)
        if required:
            experiment_id, tool = next(iter(required.items()))
            return self._required_action_turn(experiment_id, tool, source_state)

        if not transcript:
            if source_state.get("phase") == "idle":
                return AgentTurn(
                    rationale="No source work is immediately available.",
                    action=Wait(seconds=60, reason="Back off before discovering again."),
                )
            if source_state.get("phase") == "active" and source_state.get("objective"):
                objective = source_state["objective"]
                completed = set(source_state.get("completed_resource_ids", []))
                experiment_ids = [
                    experiment_id
                    for experiment_id in objective.get("resource_ids", [])
                    if experiment_id not in completed
                ]
                if experiment_ids:
                    return AgentTurn(
                        rationale="Resume the durable portfolio from persisted source state.",
                        actions=[
                            GetExperiment(experiment_id=experiment_id)
                            for experiment_id in experiment_ids
                        ],
                    )
                return AgentTurn(
                    rationale="No portfolio work is immediately available.",
                    action=Wait(seconds=60, reason="Back off before checking source state again."),
                )
            return self._discovery_turn()

        latest = transcript[-1]
        tool = latest["tool"]
        if tool == "list_experiments":
            items = latest["result"]["items"]
            active = [
                item
                for item in items
                if item["id"] not in source_state.get("completed_resource_ids", [])
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
                    resource_ids=[item["id"] for item in active],
                    summary="Interleave monitoring across all active experiments.",
                ),
            )

        if tool == "select_objective":
            experiment_ids = latest["result"]["resource_ids"]
            return AgentTurn(
                rationale="Inspect the independent experiments in this portfolio concurrently.",
                actions=[
                    GetExperiment(experiment_id=experiment_id) for experiment_id in experiment_ids
                ],
            )

        if tool == "get_experiment":
            objective = source_state.get("objective") or {}
            evidence = source_state.get("evidence", {})
            ready_ids = [
                experiment_id
                for experiment_id in objective.get("resource_ids", [])
                if evidence.get(experiment_id, {}).get("alert_ready")
                and not evidence.get(experiment_id, {}).get("results_observed")
            ]
            if ready_ids:
                return AgentTurn(
                    rationale="Read all independently available result payloads concurrently.",
                    actions=[
                        ListExperimentResults(experiment_id=experiment_id)
                        for experiment_id in ready_ids
                    ],
                )
            delayed_ids = [
                experiment_id
                for experiment_id in objective.get("resource_ids", [])
                if evidence.get(experiment_id, {}).get("status") == ExperimentStatus.DONE
                and not evidence.get(experiment_id, {}).get("results_observed")
                and not evidence.get(experiment_id, {}).get("updates_observed")
            ]
            if delayed_ids:
                return AgentTurn(
                    rationale="Inspect updates for terminal experiments whose results are delayed.",
                    actions=[
                        ListExperimentUpdates(experiment_id=experiment_id)
                        for experiment_id in delayed_ids
                    ],
                )
            return AgentTurn(
                rationale="The observed portfolio still contains active or delayed work.",
                action=Wait(seconds=1, reason="Poll the active portfolio again."),
            )

        if tool == "update_experiment":
            return AgentTurn(
                rationale="The draft now satisfies policy and can be submitted.",
                action=SubmitExperiment(experiment_id=latest["call"]["experiment_id"]),
            )

        if tool == "submit_experiment":
            return AgentTurn(
                rationale="Submission succeeded; allow the experiment lifecycle to progress.",
                action=Wait(seconds=1, reason="Wait before checking submitted work."),
            )

        if tool == "get_experiment_quote":
            return AgentTurn(
                rationale="The quote is ready for an explicit operator decision.",
                action=AcceptExperimentQuote(experiment_id=latest["call"]["experiment_id"]),
            )

        if tool == "accept_experiment_quote":
            return AgentTurn(
                rationale="The approved quote was accepted; monitor the resulting work.",
                action=Wait(seconds=1, reason="Wait before checking the accepted experiment."),
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
            completed = set(source_state.get("completed_resource_ids", []))
            evidence = source_state.get("evidence", {})
            objective = source_state.get("objective") or {}
            next_ready = next(
                (
                    experiment_id
                    for experiment_id in objective.get("resource_ids", [])
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
                action=Wait(seconds=1, reason="Pause before refreshing the active portfolio."),
            )

        return self._discovery_turn()

    @staticmethod
    def _discovery_turn() -> AgentTurn:
        """Return the Foundry discovery action for a new invocation."""
        return AgentTurn(
            rationale="Discover current experiments before selecting an objective.",
            action=ListExperiments(),
        )

    @staticmethod
    def _last_focused_experiment_id(transcript: list[dict[str, Any]]) -> str | None:
        """Return the experiment most recently inspected in this invocation."""
        for entry in reversed(transcript):
            if entry["tool"] == "get_experiment":
                return str(entry["result"]["id"])
        return None

    @staticmethod
    def _required_action_turn(
        experiment_id: str, tool: str, source_state: dict[str, Any]
    ) -> AgentTurn:
        """Choose the source operation explicitly required by normalized evidence."""
        if tool == "update_experiment":
            requirement = (
                source_state.get("evidence", {})
                .get(experiment_id, {})
                .get("requirements", {})
                .get("submit_experiment", {})
            )
            return AgentTurn(
                rationale="The draft does not satisfy its structured submission requirements.",
                action=UpdateExperiment(
                    experiment_id=experiment_id,
                    changes=ModifyExperimentRequest(
                        n_replicates=int(requirement.get("minimum_replicates", 0))
                    ),
                ),
            )
        if tool == "submit_experiment":
            return AgentTurn(
                rationale="The draft now satisfies its submission requirements.",
                action=SubmitExperiment(experiment_id=experiment_id),
            )
        if tool == "get_experiment_quote":
            return AgentTurn(
                rationale="Read the quote price before requesting approval.",
                action=GetExperimentQuote(experiment_id=experiment_id),
            )
        if tool == "accept_experiment_quote":
            return AgentTurn(
                rationale="The inspected quote is ready for an explicit operator decision.",
                action=AcceptExperimentQuote(experiment_id=experiment_id),
            )
        raise ValueError(f"Unsupported required demo action: {tool}")
