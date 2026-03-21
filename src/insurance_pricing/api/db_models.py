from sqlalchemy import Column, Integer, Float, DateTime, JSON
from datetime import datetime, timezone
from .database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # On stocke les caractéristiques envoyées par le client sous forme de dictionnaire JSON
    input_data = Column(JSON, nullable=False) 
    
    # On stocke les prédictions
    prediction_freq = Column(Float, nullable=True)
    prediction_sev = Column(Float, nullable=True)
    prediction_prime = Column(Float, nullable=False)
