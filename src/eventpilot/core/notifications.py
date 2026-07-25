"""Define provider-neutral operator notification contracts."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class NotificationPriority(StrEnum):
    """Delivery urgency understood by notification providers."""

    NORMAL = "normal"
    HIGH = "high"


class Notification(BaseModel):
    """Hold a channel-neutral, evidence-grounded message ready for delivery."""

    model_config = ConfigDict(frozen=True)

    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL


class DeliveryResult(BaseModel):
    """Record the provider identity returned after successful delivery."""

    model_config = ConfigDict(frozen=True)

    channel: str
    message_id: str
