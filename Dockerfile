FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README_architecture.md ./
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY src ./src
COPY artifacts/models ./.bundled_artifacts/models

RUN uv sync --frozen --no-dev --group api

EXPOSE 8000

CMD ["sh", "-c", "uv run --no-sync insurance-pricing-api --host 0.0.0.0 --port ${PORT:-8000}"]
