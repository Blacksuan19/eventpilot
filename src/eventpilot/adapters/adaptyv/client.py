"""Define the transport-independent Adaptyv Foundry client contract."""

from typing import Protocol

from eventpilot.adapters.adaptyv.models import (
    ExperimentConfirmation,
    ExperimentPage,
    ExperimentQuote,
    FoundryExperiment,
    ModifyExperimentRequest,
    QuoteConfirmation,
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

    async def update_experiment(
        self, experiment_id: str, changes: ModifyExperimentRequest
    ) -> FoundryExperiment:
        """Modify an editable draft or in-review experiment."""
        ...

    async def submit_experiment(self, experiment_id: str) -> ExperimentConfirmation:
        """Submit a draft experiment for review and quote preparation."""
        ...

    async def accept_experiment_quote(self, experiment_id: str) -> QuoteConfirmation:
        """Accept an experiment quote and create its invoice."""
        ...

    async def get_experiment_quote(self, experiment_id: str) -> ExperimentQuote:
        """Return the current price and expiry for an experiment quote."""
        ...
