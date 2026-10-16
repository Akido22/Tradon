"""Task Scheduler (see ARCHITECTURE.md §3.6).

Thin wrapper around APScheduler's AsyncIOScheduler for periodic jobs
(REST polling on a cadence). Realtime WebSocket streams run as their own
asyncio tasks, independent of this scheduler.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler


class TaskScheduler:
    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()

    def add_periodic(
        self, func: Callable[[], Awaitable[None]], seconds: int, job_id: str, run_immediately: bool = True
    ) -> None:
        kwargs = {"next_run_time": datetime.now()} if run_immediately else {}
        self._scheduler.add_job(func, "interval", seconds=seconds, id=job_id, replace_existing=True, **kwargs)

    def reschedule(self, job_id: str, seconds: int) -> None:
        self._scheduler.reschedule_job(job_id, trigger="interval", seconds=seconds)

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self) -> None:
        # Safe to call even if start() was never reached — e.g. LiveSession
        # only starts this scheduler when it falls back to polling; calling
        # stop() on a session that stayed in realtime mode must not crash.
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
