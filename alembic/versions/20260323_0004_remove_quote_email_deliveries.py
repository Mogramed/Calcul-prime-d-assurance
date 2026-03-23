"""Drop legacy quote email delivery table."""

from __future__ import annotations

from alembic import op


revision = "20260323_0004"
down_revision = "20260323_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_quote_email_deliveries_user_id", table_name="quote_email_deliveries")
    op.drop_index("ix_quote_email_deliveries_quote_id", table_name="quote_email_deliveries")
    op.drop_table("quote_email_deliveries")


def downgrade() -> None:
    op.execute(
        """
        CREATE TABLE quote_email_deliveries (
            id VARCHAR(36) NOT NULL PRIMARY KEY,
            quote_id VARCHAR(36) NOT NULL REFERENCES quotes (id) ON DELETE CASCADE,
            user_id VARCHAR(36) NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            recipient_email VARCHAR(320) NOT NULL,
            delivery_status VARCHAR(32) NOT NULL,
            sent_at_utc TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
        )
        """
    )
    op.create_index(
        "ix_quote_email_deliveries_quote_id",
        "quote_email_deliveries",
        ["quote_id"],
        unique=False,
    )
    op.create_index(
        "ix_quote_email_deliveries_user_id",
        "quote_email_deliveries",
        ["user_id"],
        unique=False,
    )
