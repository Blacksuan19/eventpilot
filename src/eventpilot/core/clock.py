"""Provide source-neutral clocks for time-compressed integration runs."""

import asyncio
from time import time


class AcceleratedClock:
    """Advance logical runtime time faster than physical Docker test time."""

    def __init__(self, acceleration: float) -> None:
        """Start at wall-clock time with a positive logical-time multiplier."""
        if acceleration <= 0:
            raise ValueError("Time acceleration must be positive")
        self._now = time()
        self._acceleration = acceleration

    def __call__(self) -> float:
        """Return the current logical timestamp shared by runtime components."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Advance logical time and yield briefly without a long physical sleep."""
        self._now += seconds * self._acceleration
        await asyncio.sleep(min(seconds, 0.05))
