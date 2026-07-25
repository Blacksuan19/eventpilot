"""Define human-approval values exchanged through LangGraph interrupts."""

from dataclasses import dataclass
from enum import StrEnum


class ApprovalDecision(StrEnum):
    """Represent the operator's response to a pending action."""

    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Describe a source action that requires an operator decision."""

    title: str
    body: str
    resource_ids: tuple[str, ...]
