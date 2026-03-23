"""Create public quote history table."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260323_0002"
down_revision = "20260320_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quotes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("client_id_hash", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("input_payload_json", sa.JSON(), nullable=False),
        sa.Column("frequency_prediction", sa.Float(), nullable=False),
        sa.Column("severity_prediction", sa.Float(), nullable=False),
        sa.Column("prime_prediction", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_quotes_client_id_hash", "quotes", ["client_id_hash"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_quotes_client_id_hash", table_name="quotes")
    op.drop_table("quotes")
