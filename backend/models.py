from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    text = Column(String, nullable=False)
    label = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Prediction(user_id={self.user_id}, label={self.label}, score={self.score})>"
