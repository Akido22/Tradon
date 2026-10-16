"""ORM models — minimal MVP schema (see ARCHITECTURE.md §3.5).

Rules baked in from day one so we never have to fight the database later:
- prices/volumes are Numeric, never Float;
- all timestamps are timezone-aware (UTC), never naive;
- strategy params are JSON;
- candles are indexed on (symbol, timeframe, timestamp).

Uses only generic SQLAlchemy types (no `postgresql.JSONB` etc.) so the same
models work unchanged on SQLite (default, see tradon.config) or Postgres
(docker-compose.yml) — swapping backends is a `DATABASE_URL` change only.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Numeric, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


NUMERIC = Numeric(precision=18, scale=8)
TZ_TIMESTAMP = DateTime(timezone=True)


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(unique=True, index=True)
    market_type: Mapped[str]  # e.g. "forex", "crypto"
    exchange: Mapped[str]


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        Index("ix_candles_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
        UniqueConstraint("symbol", "timeframe", "timestamp", "source", name="uq_candles_identity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str]
    timeframe: Mapped[str]
    open: Mapped[Decimal] = mapped_column(NUMERIC)
    high: Mapped[Decimal] = mapped_column(NUMERIC)
    low: Mapped[Decimal] = mapped_column(NUMERIC)
    close: Mapped[Decimal] = mapped_column(NUMERIC)
    volume: Mapped[Decimal] = mapped_column(NUMERIC)
    timestamp: Mapped[datetime] = mapped_column(TZ_TIMESTAMP, index=True)
    source: Mapped[str]


class StrategyConfig(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(default=True)


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"))
    symbol: Mapped[str]
    direction: Mapped[str]
    timestamp: Mapped[datetime] = mapped_column(TZ_TIMESTAMP, index=True)
    strength: Mapped[Decimal] = mapped_column(Numeric(precision=5, scale=4))
    status: Mapped[str] = mapped_column(default="new")
    reason: Mapped[str | None] = mapped_column(nullable=True)
    # RiskManager's suggestion at signal time — nullable because ATR needs
    # more history than a strategy itself does, so it can be unavailable
    # even when a signal fires. Never a real order (signal-only, §0/§6).
    size: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    entry: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    stop_loss: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    take_profit: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)


class Trade(Base):
    """Schema reserved from day one; populated starting Stage 6 (auto-execution)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"))
    entry: Mapped[Decimal] = mapped_column(NUMERIC)
    exit: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    size: Mapped[Decimal] = mapped_column(NUMERIC)
    pnl: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)


class User(Base):
    """Reserved for Web/messenger stages (3-5)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    hashed_password: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)
