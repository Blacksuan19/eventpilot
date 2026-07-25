"""Provide source-neutral clocks for time-compressed integration runs."""

import asyncio
from time import time


class AcceleratedClock:
    """Advance logical runtime time faster than physical Docker test time."""

    def __init__(self, acceleration: float, *, max_physical_wait_seconds: float = 5) -> None:
        """Start a logical clock with a ceiling for active physical waits."""
        if acceleration <= 0:
            raise ValueError("Time acceleration must be positive")
        if max_physical_wait_seconds < 0:
            raise ValueError("Maximum physical wait must not be negative")
        self._now = time()
        self._acceleration = acceleration
        self._max_physical_wait_seconds = max_physical_wait_seconds

    def __call__(self) -> float:
        """Return the current logical timestamp shared by runtime components."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Advance logical time while retaining a bounded real process backoff."""
        await self._advance_and_sleep(seconds, min(seconds, self._max_physical_wait_seconds))

    async def sleep_unbounded(self, seconds: float) -> None:
        """Advance logical time while honoring the full physical idle interval."""
        await self._advance_and_sleep(seconds, seconds)

    async def _advance_and_sleep(self, logical_seconds: float, physical_seconds: float) -> None:
        """Advance fixture time and yield for the requested physical duration."""
        self._now += logical_seconds * self._acceleration
        await asyncio.sleep(physical_seconds)
