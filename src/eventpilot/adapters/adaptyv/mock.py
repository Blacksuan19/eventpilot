"""Implement the Foundry client with time-driven fixture records."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.resources import files
from time import monotonic
from typing import Self

from pydantic import BaseModel, Field, model_validator

from eventpilot.adapters.adaptyv.models import (
    ExperimentConfirmation,
    ExperimentPage,
    ExperimentQuote,
    ExperimentStatus,
    FoundryExperiment,
    FoundryExperimentSummary,
    FoundryResult,
    FoundryUpdate,
    ModifyExperimentRequest,
    QuoteConfirmation,
    ResultPage,
    ResultsStatus,
    UpdatePage,
)


class LifecycleStep(BaseModel):
    """Keep one hidden mock status active for a configured number of seconds."""

    status: ExperimentStatus
    duration_seconds: int = Field(ge=0)


class ExperimentScenario(BaseModel):
    """Describe one experiment and the API snapshots it exposes over time."""

    experiment: FoundryExperiment
    lifecycle: list[LifecycleStep] = Field(min_length=1)
    updates: list[FoundryUpdate] = Field(default_factory=list)
    results: list[FoundryResult] = Field(default_factory=list)
    quote: ExperimentQuote | None = None

    @model_validator(mode="after")
    def validate_related_records(self) -> Self:
        """Reject fixture records attached to a different experiment identifier."""
        experiment_id = self.experiment.id
        if any(update.experiment_id != experiment_id for update in self.updates):
            raise ValueError("Every update must belong to its scenario experiment")
        if any(result.experiment_id != experiment_id for result in self.results):
            raise ValueError("Every result must belong to its scenario experiment")
        if self.lifecycle[-1].status != ExperimentStatus.DONE:
            raise ValueError("Every mock lifecycle must end in Done")
        if self.experiment.status == ExperimentStatus.QUOTE_SENT and self.quote is None:
            raise ValueError("A quoted mock experiment must include quote details")
        return self


class MockFoundryClient:
    """Serve fixture records through the same protocol as a live Foundry client."""

    def __init__(
        self,
        scenarios: list[ExperimentScenario],
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Start every fixture lifecycle against an injectable monotonic clock."""
        if not scenarios:
            raise ValueError("At least one mock experiment is required")
        self._scenarios = {scenario.experiment.id: scenario for scenario in scenarios}
        if len(self._scenarios) != len(scenarios):
            raise ValueError("Mock experiment identifiers must be unique")
        self._clock = clock
        self._started_at = clock()
        self._started_by_experiment = {
            experiment_id: (
                None
                if scenario.experiment.status
                in {ExperimentStatus.DRAFT, ExperimentStatus.QUOTE_SENT}
                else self._started_at
            )
            for experiment_id, scenario in self._scenarios.items()
        }
        self._experiment_overrides: dict[str, FoundryExperiment] = {}
        self.inspected_ids: list[str] = []

    @classmethod
    def from_fixture(cls, *, clock: Callable[[], float] = monotonic) -> Self:
        """Load the default mock experiment collection packaged with EventPilot."""
        return cls(load_scenarios(), clock=clock)

    async def list_experiments(self, *, limit: int = 50, offset: int = 0) -> ExperimentPage:
        """Return a paginated snapshot of every experiment in the mock organization."""
        summaries = [
            FoundryExperimentSummary.model_validate(
                self._current_experiment(experiment_id).model_dump(mode="json")
            )
            for experiment_id in self._scenarios
        ]
        items = summaries[offset : offset + limit]
        return ExperimentPage(items=items, total=len(summaries), count=len(items), offset=offset)

    async def get_experiment(self, experiment_id: str) -> FoundryExperiment:
        """Return the selected experiment without changing its time-based lifecycle."""
        experiment = self._current_experiment(experiment_id)
        self.inspected_ids.append(experiment_id)
        return experiment

    async def list_experiment_updates(self, experiment_id: str) -> UpdatePage:
        """Return progress events belonging to the selected experiment."""
        scenario = self._scenario(experiment_id)
        return UpdatePage(
            items=scenario.updates,
            total=len(scenario.updates),
            count=len(scenario.updates),
            offset=0,
        )

    async def list_experiment_results(self, experiment_id: str) -> ResultPage:
        """Return results only after their configured lifecycle snapshot is available."""
        scenario = self._scenario(experiment_id)
        experiment = self._current_experiment(experiment_id)
        items = scenario.results if experiment.results_status != ResultsStatus.NONE else []
        return ResultPage(items=items, total=len(items), count=len(items), offset=0)

    async def update_experiment(
        self, experiment_id: str, changes: ModifyExperimentRequest
    ) -> FoundryExperiment:
        """Apply documented editable fields to a draft or in-review fixture."""
        experiment = self._current_experiment(experiment_id)
        if experiment.status not in {ExperimentStatus.DRAFT, ExperimentStatus.IN_REVIEW}:
            raise ValueError(f"Experiment {experiment_id} is not editable")
        payload = changes.model_dump(exclude_unset=True, exclude_none=True)
        name = payload.pop("name", experiment.name)
        spec = {**experiment.experiment_spec, **payload}
        updated = experiment.model_copy(update={"name": name, "experiment_spec": spec})
        self._experiment_overrides[experiment_id] = updated
        return updated

    async def submit_experiment(self, experiment_id: str) -> ExperimentConfirmation:
        """Advance a draft fixture into its configured post-submission lifecycle."""
        experiment = self._current_experiment(experiment_id)
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment {experiment_id} is not a draft")
        self._activate_after_initial_status(experiment_id)
        submitted = self._current_experiment(experiment_id)
        return ExperimentConfirmation(
            experiment_id=experiment_id,
            previous_status=ExperimentStatus.DRAFT,
            status=submitted.status,
            confirmed_at=datetime.now(UTC),
        )

    async def accept_experiment_quote(self, experiment_id: str) -> QuoteConfirmation:
        """Accept a quoted fixture and expose a synthetic invoice reference."""
        experiment = self._current_experiment(experiment_id)
        if experiment.status != ExperimentStatus.QUOTE_SENT:
            raise ValueError(f"Experiment {experiment_id} has no pending quote")
        self._activate_after_initial_status(experiment_id)
        slug = experiment_id.removeprefix("experiment-")
        return QuoteConfirmation(
            id=f"quote-{slug}",
            status="accepted",
            hosted_invoice_url=f"https://invoice.stripe.test/{slug}",
            invoice_id=f"invoice-{slug}",
        )

    async def get_experiment_quote(self, experiment_id: str) -> ExperimentQuote:
        """Return configured quote details for a quoted fixture experiment."""
        experiment = self._current_experiment(experiment_id)
        quote = self._scenario(experiment_id).quote
        if experiment.status != ExperimentStatus.QUOTE_SENT or quote is None:
            raise ValueError(f"Experiment {experiment_id} has no pending quote")
        return quote

    def _current_experiment(self, experiment_id: str) -> FoundryExperiment:
        """Render status from hidden elapsed time over immutable fixture details."""
        scenario = self._scenario(experiment_id)
        started_at = self._started_by_experiment[experiment_id]
        base = self._experiment_overrides.get(experiment_id, scenario.experiment)
        if started_at is None:
            return base.model_copy(
                update={"status": scenario.experiment.status, "results_status": ResultsStatus.NONE}
            )
        elapsed = self._clock() - started_at
        remaining = elapsed
        status = scenario.lifecycle[-1].status
        for step in scenario.lifecycle:
            status = step.status
            if remaining < step.duration_seconds:
                break
            remaining -= step.duration_seconds
        total_duration = sum(step.duration_seconds for step in scenario.lifecycle)
        results_available = bool(scenario.results) and elapsed >= total_duration
        return base.model_copy(
            update={
                "status": status,
                "results_status": ResultsStatus.ALL if results_available else ResultsStatus.NONE,
            }
        )

    def _activate_after_initial_status(self, experiment_id: str) -> None:
        """Start a gated lifecycle immediately after its draft or quote step."""
        scenario = self._scenario(experiment_id)
        initial_duration = scenario.lifecycle[0].duration_seconds
        self._started_by_experiment[experiment_id] = self._clock() - initial_duration

    def _scenario(self, experiment_id: str) -> ExperimentScenario:
        """Resolve one fixture scenario or match the API's not-found boundary."""
        try:
            return self._scenarios[experiment_id]
        except KeyError as exc:
            raise LookupError(f"Mock Foundry experiment not found: {experiment_id}") from exc


def load_scenarios() -> list[ExperimentScenario]:
    """Parse a fresh copy of every packaged experiment scenario."""
    fixture = files("eventpilot.adapters.adaptyv.fixtures").joinpath("experiments.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    return [ExperimentScenario.model_validate(item) for item in payload]
