"""initial schema

Revision ID: 3058d7aecaf7
Revises:
Create Date: 2026-08-12

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3058d7aecaf7"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

NUMERIC = sa.Numeric(precision=18, scale=8)
TZ_TIMESTAMP = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("market_type", sa.String(), nullable=False),
        sa.Column("exchange", sa.String(), nullable=False),
        sa.UniqueConstraint("symbol", name="uq_instruments_symbol"),
    )
    op.create_index("ix_instruments_symbol", "instruments", ["symbol"])

    op.create_table(
        "candles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("timeframe", sa.String(), nullable=False),
        sa.Column("open", NUMERIC, nullable=False),
        sa.Column("high", NUMERIC, nullable=False),
        sa.Column("low", NUMERIC, nullable=False),
        sa.Column("close", NUMERIC, nullable=False),
        sa.Column("volume", NUMERIC, nullable=False),
        sa.Column("timestamp", TZ_TIMESTAMP, nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.UniqueConstraint("symbol", "timeframe", "timestamp", "source", name="uq_candles_identity"),
    )
    op.create_index("ix_candles_timestamp", "candles", ["timestamp"])
    op.create_index(
        "ix_candles_symbol_timeframe_timestamp", "candles", ["symbol", "timeframe", "timestamp"]
    )

    op.create_table(
        "strategies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("name", name="uq_strategies_name"),
    )

    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("strategy_id", sa.Integer(), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("timestamp", TZ_TIMESTAMP, nullable=False),
        sa.Column("strength", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="new"),
    )
    op.create_index("ix_signals_timestamp", "signals", ["timestamp"])

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("signal_id", sa.Integer(), sa.ForeignKey("signals.id"), nullable=False),
        sa.Column("entry", NUMERIC, nullable=False),
        sa.Column("exit", NUMERIC, nullable=True),
        sa.Column("size", NUMERIC, nullable=False),
        sa.Column("pnl", NUMERIC, nullable=True),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )


def downgrade() -> None:
    op.drop_table("users")
    op.drop_table("trades")
    op.drop_index("ix_signals_timestamp", table_name="signals")
    op.drop_table("signals")
    op.drop_table("strategies")
    op.drop_index("ix_candles_symbol_timeframe_timestamp", table_name="candles")
    op.drop_index("ix_candles_timestamp", table_name="candles")
    op.drop_table("candles")
    op.drop_index("ix_instruments_symbol", table_name="instruments")
    op.drop_table("instruments")
