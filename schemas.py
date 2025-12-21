from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List

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

class TrainingRequest(BaseModel):
    dataset_path: str
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4

class TrainingStatusResponse(BaseModel):
    is_training: bool
    progress: int
    status: str
    message: str
    history: List[Dict]
