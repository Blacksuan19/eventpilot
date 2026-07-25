"""Model documented Adaptyv Foundry API responses."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentStatus(StrEnum):
    """Lifecycle values documented by the Foundry experiments API."""

    DRAFT = "Draft"
    WAITING_FOR_CONFIRMATION = "WaitingForConfirmation"
    QUOTE_SENT = "QuoteSent"
    WAITING_FOR_MATERIALS = "WaitingForMaterials"
    IN_QUEUE = "InQueue"
    IN_PRODUCTION = "InProduction"
    DATA_ANALYSIS = "DataAnalysis"
    IN_REVIEW = "InReview"
    DONE = "Done"
    CANCELED = "Canceled"


class ResultsStatus(StrEnum):
    """Availability values documented by the Foundry experiments API."""

    NONE = "None"
    PARTIAL = "Partial"
    ALL = "All"


class FoundryExperimentSummary(BaseModel):
    """Represent an item returned by `GET /api/v1/experiments`."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    status: ExperimentStatus
    created_at: datetime
    results_status: ResultsStatus
    experiment_url: str
    name: str | None = None
    experiment_type: str | None = None


class FoundryExperiment(FoundryExperimentSummary):
    """Represent `GET /api/v1/experiments/{experiment_id}`."""

    experiment_spec: dict[str, Any]


class ModifyExperimentRequest(BaseModel):
    """Represent documented editable fields accepted by the Foundry API."""

    name: str | None = None
    description: str | None = None
    n_replicates: int | None = Field(default=None, ge=0)
    parameters: dict[str, Any] | None = None
    target_id: str | None = None
    webhook_url: str | None = None

    @model_validator(mode="after")
    def require_change(self) -> "ModifyExperimentRequest":
        """Reject an update request that contains no fields to modify."""
        if not self.model_fields_set:
            raise ValueError("At least one experiment field must be provided")
        return self


class ExperimentConfirmation(BaseModel):
    """Represent the status transition returned after draft submission."""

    experiment_id: str
    previous_status: ExperimentStatus
    status: ExperimentStatus
    confirmed_at: datetime
    stripe_invoice_url: str | None = None


class QuoteConfirmation(BaseModel):
    """Represent an accepted quote and its generated invoice reference."""

    id: str
    status: str
    hosted_invoice_url: str | None = None
    invoice_id: str | None = None


class ExperimentQuote(BaseModel):
    """Represent billing details returned for one experiment quote."""

    experiment_id: str
    stripe_quote_url: str
    amount_total: int = Field(ge=0)
    amount_subtotal: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    status: str
    expires_at: datetime | None = None
    updated_at: datetime | None = None


class FoundryUpdate(BaseModel):
    """Represent an update returned by an experiment's updates endpoint."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    experiment_code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    timestamp: datetime


class FoundryResult(BaseModel):
    """Represent an analysis result returned for an experiment."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    result_type: str = Field(min_length=1)
    created_at: datetime
    summary: dict[str, Any]
    metadata: dict[str, Any]
    data_package_url: str | None = None


class ExperimentPage(BaseModel):
    """Validate the documented paginated experiment-list response."""

    items: list[FoundryExperimentSummary]
    total: int = Field(ge=0)
    count: int = Field(ge=0)
    offset: int = Field(ge=0)


class UpdatePage(BaseModel):
    """Validate the documented paginated experiment-update response."""

    items: list[FoundryUpdate]
    total: int = Field(ge=0)
    count: int = Field(ge=0)
    offset: int = Field(ge=0)


class ResultPage(BaseModel):
    """Validate the documented paginated experiment-results response."""

    items: list[FoundryResult]
    total: int = Field(ge=0)
    count: int = Field(ge=0)
    offset: int = Field(ge=0)
