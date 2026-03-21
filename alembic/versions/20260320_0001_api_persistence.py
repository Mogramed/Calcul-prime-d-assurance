"""Create API persistence tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260320_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_requests",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=255), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("endpoint", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prediction_requests_request_id",
        "prediction_requests",
        ["request_id"],
        unique=False,
    )

    op.create_table(
        "prediction_outputs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("prediction_request_id", sa.String(length=36), nullable=False),
        sa.Column("record_position", sa.Integer(), nullable=False),
        sa.Column("input_index", sa.Integer(), nullable=True),
        sa.Column("frequency_prediction", sa.Float(), nullable=True),
        sa.Column("severity_prediction", sa.Float(), nullable=True),
        sa.Column("prime_prediction", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["prediction_request_id"],
            ["prediction_requests.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prediction_outputs_prediction_request_id",
        "prediction_outputs",
        ["prediction_request_id"],
        unique=False,
    )

    op.create_table(
        "api_errors",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=255), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("endpoint", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("exception_type", sa.String(length=255), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("traceback_excerpt", sa.Text(), nullable=True),
        sa.Column("payload_hash", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_errors_request_id", "api_errors", ["request_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_api_errors_request_id", table_name="api_errors")
    op.drop_table("api_errors")
    op.drop_index("ix_prediction_outputs_prediction_request_id", table_name="prediction_outputs")
    op.drop_table("prediction_outputs")
    op.drop_index("ix_prediction_requests_request_id", table_name="prediction_requests")
    op.drop_table("prediction_requests")
