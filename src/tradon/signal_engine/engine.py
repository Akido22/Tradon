"""Signal Engine (see ARCHITECTURE.md §3.3).

Aggregates raw signals coming from strategy plugins, filters/dedups them,
and publishes confirmed signals on an internal event bus that CLI/DB/
notification subscribers listen to.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import timedelta

from tradon.strategy.base import Signal, SignalDirection


class SignalEngine:
    def __init__(self, min_confirmations: int = 1, confirmation_window: timedelta = timedelta(minutes=5)) -> None:
        self.min_confirmations = min_confirmations
        self.confirmation_window = confirmation_window
        self._subscribers: list[asyncio.Queue[Signal]] = []
        self._recent: dict[tuple[str, SignalDirection], list[Signal]] = defaultdict(list)

    def subscribe(self) -> asyncio.Queue[Signal]:
        """Register a new subscriber (CLI, DB writer, notifier, ...) and get its queue."""
        queue: asyncio.Queue[Signal] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    async def ingest(self, signal: Signal) -> Signal | None:
        """Feed a raw strategy signal in. Returns the published signal if it clears the filters."""
        key = (signal.symbol, signal.direction)
        bucket = self._recent[key]
        bucket[:] = [s for s in bucket if signal.timestamp - s.timestamp <= self.confirmation_window]

        if any(s.strategy == signal.strategy for s in bucket):
            return None  # dedup: same strategy already signalled this direction recently

        bucket.append(signal)
        if len(bucket) < self.min_confirmations:
            return None

        await self._publish(signal)
        return signal

    async def _publish(self, signal: Signal) -> None:
        for queue in self._subscribers:
            await queue.put(signal)
