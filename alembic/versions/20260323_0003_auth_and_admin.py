"""Add users, sessions, quote moderation, and email delivery tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260323_0003"
down_revision = "20260323_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_auth_sessions_token_hash", "auth_sessions", ["token_hash"], unique=True)
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"], unique=False)

    op.add_column("quotes", sa.Column("user_id", sa.String(length=36), nullable=True))
    op.add_column("quotes", sa.Column("deleted_at_utc", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_quotes_user_id", "quotes", ["user_id"], unique=False)
    op.create_foreign_key(
        "fk_quotes_user_id_users",
        "quotes",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "quote_email_deliveries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("quote_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("recipient_email", sa.String(length=320), nullable=False),
        sa.Column("delivery_status", sa.String(length=32), nullable=False),
        sa.Column("sent_at_utc", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["quote_id"], ["quotes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_quote_email_deliveries_quote_id", "quote_email_deliveries", ["quote_id"], unique=False)
    op.create_index("ix_quote_email_deliveries_user_id", "quote_email_deliveries", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_quote_email_deliveries_user_id", table_name="quote_email_deliveries")
    op.drop_index("ix_quote_email_deliveries_quote_id", table_name="quote_email_deliveries")
    op.drop_table("quote_email_deliveries")

    op.drop_constraint("fk_quotes_user_id_users", "quotes", type_="foreignkey")
    op.drop_index("ix_quotes_user_id", table_name="quotes")
    op.drop_column("quotes", "deleted_at_utc")
    op.drop_column("quotes", "user_id")

    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_token_hash", table_name="auth_sessions")
    op.drop_table("auth_sessions")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
