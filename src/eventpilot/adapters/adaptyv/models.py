"""Model documented Adaptyv Foundry API responses."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
