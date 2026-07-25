"""Define the transport-independent Adaptyv Foundry client contract."""

from typing import Protocol

from eventpilot.adapters.adaptyv.models import (
    ExperimentPage,
    FoundryExperiment,
    ResultPage,
    UpdatePage,
)


class FoundryClient(Protocol):
    """Expose only documented Foundry operations available to a data source."""

    async def list_experiments(self, *, limit: int = 50, offset: int = 0) -> ExperimentPage:
        """List experiments visible to the authenticated organization."""
        ...

    async def get_experiment(self, experiment_id: str) -> FoundryExperiment:
        """Fetch the current detailed representation of one experiment."""
        ...

    async def list_experiment_updates(self, experiment_id: str) -> UpdatePage:
        """List chronological updates for one experiment."""
        ...

    async def list_experiment_results(self, experiment_id: str) -> ResultPage:
        """List available analysis results for one experiment."""
        ...
