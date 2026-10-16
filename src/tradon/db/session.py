"""Async SQLAlchemy engine/session factory — the only DB access path (§3.5)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry
from sqlite3 import Connection as SQLite3Connection

from tradon.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    url = get_settings().database_url
    if url.startswith("sqlite"):
        # sqlite:///relative/path resolves against the process cwd, not the
        # project root, so the data dir must exist before connecting.
        path = url.split("///", 1)[1]
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(url, pool_pre_ping=True)

    if url.startswith("sqlite"):
        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection: SQLite3Connection, _: ConnectionPoolEntry) -> None:
            # WAL: readers never block on a writer (CLI + GUI open at once).
            # busy_timeout: a second writer retries for 5s instead of
            # failing immediately with "database is locked".
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session
