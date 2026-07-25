"""Provide an API-compatible Foundry simulator for local demonstrations."""

from datetime import UTC, datetime
from typing import Any

from eventpilot.adapters.adaptyv import (
    ExperimentPage,
    FoundryExperiment,
    FoundryExperimentSummary,
    FoundryResult,
    FoundryUpdate,
    ResultPage,
    UpdatePage,
)

EXPERIMENT_ID = "019d4a2b-2b7e-7c3a-9f1e-2a4b6c8d0e1f"
RESULT_ID = "019d4a2c-3c8f-7d4b-a02f-3b5c7d9e1f20"


def mock_experiment(*, status: str = "Done", results_status: str = "All") -> FoundryExperiment:
    """Build one fixture matching the documented Foundry detail response."""
    return FoundryExperiment.model_validate(
        {
            "id": EXPERIMENT_ID,
            "code": "ORG-001-123",
            "name": "EGFR Binding Screen",
            "status": status,
            "experiment_type": "screening",
            "experiment_spec": {"experiment_type": "screening", "method": "bli"},
            "created_at": "2026-07-01T10:00:00Z",
            "results_status": results_status,
            "experiment_url": f"https://foundry.adaptyvbio.com/experiments/{EXPERIMENT_ID}",
        }
    )


class ProgressingFoundryClient:
    """Mock the documented Foundry endpoints while experiment state advances externally."""

    def __init__(self, statuses: list[str]) -> None:
        """Store a non-empty lifecycle sequence returned by successive detail calls."""
        if not statuses:
            raise ValueError("At least one mock status is required")
        self._statuses = statuses
        self._index = 0

    async def list_experiments(self, *, limit: int = 50, offset: int = 0) -> ExperimentPage:
        """Return the documented paginated list shape without advancing external state."""
        experiment = self._current_experiment()
        summary = FoundryExperimentSummary.model_validate(experiment.model_dump(mode="json"))
        items = [summary][offset : offset + limit]
        return ExperimentPage(items=items, total=1, count=len(items), offset=offset)

    async def get_experiment(self, experiment_id: str) -> FoundryExperiment:
        """Return the next detailed snapshot as if Foundry changed between polls."""
        self._require_experiment(experiment_id)
        experiment = self._current_experiment()
        self._index += 1
        return experiment

    async def list_experiment_updates(self, experiment_id: str) -> UpdatePage:
        """Return one documented progress-update record for the experiment."""
        self._require_experiment(experiment_id)
        item = FoundryUpdate(
            id="019d4a2d-4d90-7e5c-b13f-4c6d8e0f2a31",
            experiment_id=EXPERIMENT_ID,
            experiment_code="ORG-001-123",
            name="Experiment processing update",
            timestamp=datetime.now(UTC),
        )
        return UpdatePage(items=[item], total=1, count=1, offset=0)

    async def list_experiment_results(self, experiment_id: str) -> ResultPage:
        """Return a documented result page once the mock experiment is complete."""
        self._require_experiment(experiment_id)
        experiment = self._current_experiment()
        if experiment.results_status == "None":
            return ResultPage(items=[], total=0, count=0, offset=0)
        item = FoundryResult(
            id=RESULT_ID,
            title="EGFR binding screen",
            experiment_id=EXPERIMENT_ID,
            result_type="screening",
            created_at=datetime.now(UTC),
            summary={"classification": "binder"},
            metadata={},
            data_package_url="https://foundry.adaptyvbio.com/results/package.zip",
        )
        return ResultPage(items=[item], total=1, count=1, offset=0)

    def _current_experiment(self) -> FoundryExperiment:
        """Build the snapshot at the mock API's current external-state index."""
        status = self._statuses[min(self._index, len(self._statuses) - 1)]
        return mock_experiment(
            status=status,
            results_status="All" if status == "Done" else "None",
        )

    @staticmethod
    def _require_experiment(experiment_id: str) -> None:
        """Raise the same not-found boundary for unsupported mock identifiers."""
        if experiment_id != EXPERIMENT_ID:
            raise LookupError(f"Mock Foundry experiment not found: {experiment_id}")


def tool_names(transcript: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from an agent transcript for readable trajectory assertions."""
    return [str(entry["tool"]) for entry in transcript]
