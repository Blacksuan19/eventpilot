"""Expose documented Foundry operations as typed agent tools."""

from typing import Literal

from pydantic import Field

from eventpilot.adapters.adaptyv.models import ModifyExperimentRequest
from eventpilot.sources.base import SourceToolCall, ToolAvailability


class ListExperiments(SourceToolCall):
    """List experiments visible to the authenticated Foundry organization."""

    tool: Literal["list_experiments"] = "list_experiments"
    limit: int = Field(default=50, ge=1, le=100, description="Maximum records to return.")
    offset: int = Field(default=0, ge=0, description="Zero-based pagination offset.")


class GetExperiment(SourceToolCall):
    """Retrieve the current detailed representation of one Foundry experiment."""

    parallel_safe = True
    tool: Literal["get_experiment"] = "get_experiment"
    experiment_id: str = Field(min_length=1, description="Foundry experiment identifier.")


class ListExperimentUpdates(SourceToolCall):
    """List chronological progress updates for one Foundry experiment."""

    parallel_safe = True
    tool: Literal["list_experiment_updates"] = "list_experiment_updates"
    experiment_id: str = Field(min_length=1, description="Foundry experiment identifier.")


class ListExperimentResults(SourceToolCall):
    """List analysis results currently available for one Foundry experiment."""

    parallel_safe = True
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
    availability = ToolAvailability(
        statuses=("Draft",),
        evidence_keys=("detail_observed",),
        requirement_key="submit_experiment",
    )
    experiment_id: str = Field(min_length=1, description="Draft experiment identifier.")


class AcceptExperimentQuote(SourceToolCall):
    """Accept a Foundry quote and create an invoice after operator approval."""

    tool: Literal["accept_experiment_quote"] = "accept_experiment_quote"
    availability = ToolAvailability(statuses=("QuoteSent",), evidence_keys=("quote_observed",))
    experiment_id: str = Field(min_length=1, description="Quoted experiment identifier.")


class GetExperimentQuote(SourceToolCall):
    """Retrieve price and expiry details for a Foundry experiment quote."""

    parallel_safe = True
    tool: Literal["get_experiment_quote"] = "get_experiment_quote"
    availability = ToolAvailability(statuses=("QuoteSent",), evidence_keys=("detail_observed",))
    experiment_id: str = Field(min_length=1, description="Quoted experiment identifier.")
