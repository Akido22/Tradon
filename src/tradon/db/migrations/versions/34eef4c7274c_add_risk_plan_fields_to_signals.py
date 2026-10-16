"""add risk plan fields to signals

Revision ID: 34eef4c7274c
Revises: 3058d7aecaf7
Create Date: 2026-08-31 19:19:07.074302

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "34eef4c7274c"
down_revision: str | None = "3058d7aecaf7"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

NUMERIC = sa.Numeric(precision=18, scale=8)


def upgrade() -> None:
    with op.batch_alter_table("signals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("size", NUMERIC, nullable=True))
        batch_op.add_column(sa.Column("entry", NUMERIC, nullable=True))
        batch_op.add_column(sa.Column("stop_loss", NUMERIC, nullable=True))
        batch_op.add_column(sa.Column("take_profit", NUMERIC, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("signals", schema=None) as batch_op:
        batch_op.drop_column("take_profit")
        batch_op.drop_column("stop_loss")
        batch_op.drop_column("entry")
        batch_op.drop_column("size")
