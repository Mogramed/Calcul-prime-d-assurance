"""Drop legacy input_index from prediction audit outputs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260411_0005"
down_revision = "20260323_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("prediction_outputs", "input_index")


def downgrade() -> None:
    op.add_column("prediction_outputs", sa.Column("input_index", sa.Integer(), nullable=True))
