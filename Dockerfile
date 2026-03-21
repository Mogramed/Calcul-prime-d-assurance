# On part d'une image Python officielle, légère et récente
FROM python:3.11-slim

# On définit le dossier de travail dans le conteneur
WORKDIR /app

# Installation de 'uv' pour gérer les dépendances
RUN pip install uv

# On copie d'abord les fichiers de dépendances (optimisation du cache Docker)
COPY pyproject.toml uv.lock ./

# On installe les dépendances au niveau du système du conteneur
RUN uv pip install --system -r pyproject.toml

# On copie le code source et les modèles entraînés
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# On expose le port sur lequel l'API va tourner
EXPOSE 8000

# La commande pour démarrer le serveur FastAPI
CMD ["uvicorn", "src.insurance_pricing.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
