# On part d'une image Python officielle
FROM python:3.11-slim

# On définit le dossier de travail
WORKDIR /app

# Installation de 'uv'
RUN pip install uv

# On copie les fichiers de config et le README (nécessaire pour uv)
COPY pyproject.toml uv.lock README_architecture.md ./

# MAGIE ICI : On ajoute "--group api" pour installer FastAPI et Uvicorn
RUN uv sync --group api --no-dev

# On copie le code source et les modèles
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# On expose le port
EXPOSE 8000

# MAGIE ICI : On utilise "--factory" car ton code utilise def create_app()
CMD ["uv", "run", "uvicorn", "src.insurance_pricing.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
