"""Expose documented Foundry operations as typed agent tools."""

from importlib.resources import files
from typing import Any, Literal

from pydantic import Field

from eventpilot.adapters.adaptyv.client import FoundryClient
from eventpilot.adapters.adaptyv.models import ModifyExperimentRequest
from eventpilot.sources.base import SourceToolCall, ToolAvailability


class ListExperiments(SourceToolCall):
    """List experiments visible to the authenticated Foundry organization."""

    tool: Literal["list_experiments"] = "list_experiments"
    limit: int = Field(default=50, ge=1, le=100, description="Maximum records to return.")
    offset: int = Field(default=0, ge=0, description="Zero-based pagination offset.")


class GetExperiment(SourceToolCall):
    """Retrieve the current detailed representation of one Foundry experiment."""

    tool: Literal["get_experiment"] = "get_experiment"
    experiment_id: str = Field(min_length=1, description="Foundry experiment identifier.")


class ListExperimentUpdates(SourceToolCall):
    """List chronological progress updates for one Foundry experiment."""

    tool: Literal["list_experiment_updates"] = "list_experiment_updates"
    experiment_id: str = Field(min_length=1, description="Foundry experiment identifier.")


class ListExperimentResults(SourceToolCall):
    """List analysis results currently available for one Foundry experiment."""

    tool: Literal["list_experiment_results"] = "list_experiment_results"
    experiment_id: str = Field(min_length=1, description="Foundry experiment identifier.")


class UpdateExperiment(SourceToolCall):
    """Modify fields on an editable Foundry experiment."""

    tool: Literal["update_experiment"] = "update_experiment"
    availability = ToolAvailability(
        statuses=("Draft", "InReview"), evidence_keys=("detail_observed",)
    )
    experiment_id: str = Field(min_length=1, description="Foundry experiment identifier.")
    changes: ModifyExperimentRequest = Field(description="Editable fields and replacement values.")


class SubmitExperiment(SourceToolCall):
    """Submit a draft Foundry experiment for review and quote preparation."""

    tool: Literal["submit_experiment"] = "submit_experiment"
    availability = ToolAvailability(statuses=("Draft",), evidence_keys=("detail_observed",))
    experiment_id: str = Field(min_length=1, description="Draft experiment identifier.")


class AcceptExperimentQuote(SourceToolCall):
    """Accept a Foundry quote and create an invoice after operator approval."""

    tool: Literal["accept_experiment_quote"] = "accept_experiment_quote"
    availability = ToolAvailability(statuses=("QuoteSent",), evidence_keys=("quote_observed",))
    experiment_id: str = Field(min_length=1, description="Quoted experiment identifier.")


class GetExperimentQuote(SourceToolCall):
    """Retrieve price and expiry details for a Foundry experiment quote."""

    tool: Literal["get_experiment_quote"] = "get_experiment_quote"
    availability = ToolAvailability(statuses=("QuoteSent",), evidence_keys=("detail_observed",))
    experiment_id: str = Field(min_length=1, description="Quoted experiment identifier.")


class FoundryToolAdapter:
    """Publish typed Foundry operations and execute them through a client transport."""

    tool_types: tuple[type[SourceToolCall], ...] = (
        ListExperiments,
        GetExperiment,
        ListExperimentUpdates,
        ListExperimentResults,
        UpdateExperiment,
        SubmitExperiment,
        AcceptExperimentQuote,
        GetExperimentQuote,
    )

    def __init__(self, client: FoundryClient) -> None:
        """Bind the tool adapter to a live or fixture-backed Foundry client."""
        self._client = client
        self.instructions = (
            files("eventpilot.adapters.adaptyv")
            .joinpath("instructions.txt")
            .read_text(encoding="utf-8")
        )
        self._tool_types = {
            tool_type.model_fields["tool"].default: tool_type for tool_type in self.tool_types
        }

    def parse_tool(self, payload: dict[str, Any]) -> SourceToolCall:
        """Validate a persisted operation against the adapter's published schemas."""
        tool_type = self._tool_types.get(payload.get("tool"))
        if tool_type is None:
            raise ValueError(f"Unknown Foundry adapter tool: {payload.get('tool')}")
        return tool_type.model_validate(payload)

    async def execute(self, action: SourceToolCall) -> dict[str, Any]:
        """Execute one adapter-owned tool and return its validated JSON response."""
        if isinstance(action, ListExperiments):
            page = await self._client.list_experiments(limit=action.limit, offset=action.offset)
            return page.model_dump(mode="json")
        if isinstance(action, GetExperiment):
            experiment = await self._client.get_experiment(action.experiment_id)
            return experiment.model_dump(mode="json")
        if isinstance(action, ListExperimentUpdates):
            page = await self._client.list_experiment_updates(action.experiment_id)
            return page.model_dump(mode="json")
        if isinstance(action, ListExperimentResults):
            page = await self._client.list_experiment_results(action.experiment_id)
            return page.model_dump(mode="json")
        if isinstance(action, UpdateExperiment):
            experiment = await self._client.update_experiment(action.experiment_id, action.changes)
            return experiment.model_dump(mode="json")
        if isinstance(action, SubmitExperiment):
            confirmation = await self._client.submit_experiment(action.experiment_id)
            return confirmation.model_dump(mode="json")
        if isinstance(action, AcceptExperimentQuote):
            confirmation = await self._client.accept_experiment_quote(action.experiment_id)
            return confirmation.model_dump(mode="json")
        if isinstance(action, GetExperimentQuote):
            quote = await self._client.get_experiment_quote(action.experiment_id)
            return quote.model_dump(mode="json")
        raise TypeError(f"Unsupported Foundry adapter action: {type(action).__name__}")
