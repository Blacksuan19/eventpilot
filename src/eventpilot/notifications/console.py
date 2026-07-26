"""Deliver operator updates to the process console."""

from uuid import uuid4

from eventpilot.core.notifications import DeliveryResult, Notification


class ConsoleNotificationSink:
    """Print notifications locally for demos and development."""

    channel_name = "console"

    def __init__(self) -> None:
        """Create an in-process delivery cache keyed by graph operation identity."""
        self._deliveries: dict[str, DeliveryResult] = {}

    async def send(
        self,
        destination: str,
        notification: Notification,
        *,
        idempotency_key: str,
    ) -> DeliveryResult:
        """Print a notification once and return its stable synthetic receipt."""
        if cached := self._deliveries.get(idempotency_key):
            return cached
        print(f"[{notification.priority}] {notification.title}\n{notification.body}")
        receipt = DeliveryResult(channel=self.channel_name, message_id=str(uuid4()))
        self._deliveries[idempotency_key] = receipt
        return receipt
