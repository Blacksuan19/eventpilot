"""Provide a fixture-backed mock of the Foundry API."""

import json
from collections.abc import Callable
from importlib.resources import files
from time import monotonic
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from eventpilot.adapters.adaptyv import (
    ExperimentPage,
    ExperimentStatus,
    FoundryExperiment,
    FoundryExperimentSummary,
    FoundryResult,
    FoundryUpdate,
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
        return self


class MockFoundryClient:
    """Serve any fixture collection through the same API protocol used by the agent."""

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
        self.inspected_ids: list[str] = []

    @classmethod
    def from_fixture(cls) -> Self:
        """Load the default mock experiment collection packaged with EventPilot."""
        return cls(load_scenarios())

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

    def _current_experiment(self, experiment_id: str) -> FoundryExperiment:
        """Render status from hidden elapsed time over immutable fixture details."""
        scenario = self._scenario(experiment_id)
        elapsed = self._clock() - self._started_at
        remaining = elapsed
        status = scenario.lifecycle[-1].status
        for step in scenario.lifecycle:
            status = step.status
            if remaining < step.duration_seconds:
                break
            remaining -= step.duration_seconds
        total_duration = sum(step.duration_seconds for step in scenario.lifecycle)
        results_available = bool(scenario.results) and elapsed >= total_duration
        return scenario.experiment.model_copy(
            update={
                "status": status,
                "results_status": ResultsStatus.ALL if results_available else ResultsStatus.NONE,
            }
        )

    def _scenario(self, experiment_id: str) -> ExperimentScenario:
        """Resolve one fixture scenario or match the API's not-found boundary."""
        try:
            return self._scenarios[experiment_id]
        except KeyError as exc:
            raise LookupError(f"Mock Foundry experiment not found: {experiment_id}") from exc


def tool_names(transcript: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from an agent transcript for readable trajectory assertions."""
    return [str(entry["tool"]) for entry in transcript]


def load_scenarios() -> list[ExperimentScenario]:
    """Parse a fresh copy of every packaged experiment scenario."""
    fixture = files("eventpilot.fixtures").joinpath("experiments.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    return [ExperimentScenario.model_validate(item) for item in payload]
