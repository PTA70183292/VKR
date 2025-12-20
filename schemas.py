from pydantic import BaseModel
from datetime import datetime

class PredictRequest(BaseModel):
    user_id: str
    text: str

class PredictResponse(BaseModel):
    id: int
    user_id: str
    text: str
    label: str
    score: float
    created_at: datetime
    
    class Config:
        from_attributes = True
