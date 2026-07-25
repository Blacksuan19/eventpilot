"""Deliver operator updates to the process console."""

from uuid import uuid4

from eventpilot.core.notifications import DeliveryResult, Notification


class ConsoleNotificationSink:
    """Print notifications locally for demos and development."""

    channel_name = "console"

    async def send(self, destination: str, notification: Notification) -> DeliveryResult:
        """Print a notification and return a synthetic provider message identifier."""
        print(f"[{notification.priority}] {notification.title}\n{notification.body}")
        return DeliveryResult(channel=self.channel_name, message_id=str(uuid4()))
