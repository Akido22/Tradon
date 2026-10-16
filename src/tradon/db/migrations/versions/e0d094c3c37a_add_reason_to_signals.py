"""add reason to signals

Revision ID: e0d094c3c37a
Revises: 34eef4c7274c
Create Date: 2026-09-01 09:21:10.717548

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e0d094c3c37a"
down_revision: str | None = "34eef4c7274c"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("signals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("reason", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("signals", schema=None) as batch_op:
        batch_op.drop_column("reason")
