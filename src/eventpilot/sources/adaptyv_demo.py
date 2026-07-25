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
from eventpilot.core.agent_reasoning import AgentTurn, FinishCycle, SendAlert, Wait
from eventpilot.core.monitoring import SelectObjective


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
            experiment_id = latest["result"]["resource_ids"][0]
            return AgentTurn(
                rationale="Inspect the experiment selected for this cycle.",
                action=GetExperiment(experiment_id=experiment_id),
            )

        if tool == "wait":
            if source_state.get("phase") == "discovery" and not source_state.get("objective"):
                return AgentTurn(
                    rationale="The idle backoff completed without new work.",
                    action=FinishCycle(summary="No actionable experiments after idle backoff."),
                )
            candidates = source_state.get("next_resource_candidates", [])
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
            if experiment["status"] == ExperimentStatus.DRAFT:
                replicates = experiment.get("experiment_spec", {}).get("n_replicates", 0)
                if replicates < 2:
                    return AgentTurn(
                        rationale="The draft needs at least two replicates before submission.",
                        action=UpdateExperiment(
                            experiment_id=experiment_id,
                            changes=ModifyExperimentRequest(n_replicates=3),
                        ),
                    )
                return AgentTurn(
                    rationale="The draft satisfies submission policy.",
                    action=SubmitExperiment(experiment_id=experiment_id),
                )
            if experiment["status"] == ExperimentStatus.QUOTE_SENT:
                return AgentTurn(
                    rationale="Read the quote price before requesting approval.",
                    action=GetExperimentQuote(experiment_id=experiment_id),
                )
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
