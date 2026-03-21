import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# On récupère l'URL depuis les variables d'environnement (définie dans docker-compose)
# Par défaut, si on lance sans Docker, ça créera un petit fichier SQLite local
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./insurance_local_logs.db"
)

# Configuration du moteur de base de données
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)

# Création des sessions pour interagir avec la DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Classe de base pour créer nos tables
Base = declarative_base()

# Dépendance FastAPI pour ouvrir/fermer la connexion proprement à chaque requête API
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
