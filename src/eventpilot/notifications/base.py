"""Define the protocol implemented by notification providers."""

from typing import Protocol

from eventpilot.core.notifications import DeliveryResult, Notification


class NotificationSink(Protocol):
    """Define the delivery contract implemented by trusted notification channels."""

    channel_name: str

    async def send(
        self,
        destination: str,
        notification: Notification,
        *,
        idempotency_key: str,
    ) -> DeliveryResult:
        """Deliver a notification to a preconfigured destination."""
        ...
