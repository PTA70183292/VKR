from sqlalchemy.orm import Session
from models import PredictionRecord
from typing import List, Optional

def create_prediction(
    db: Session,
    user_id: str,
    text: str,
    label: str,
    score: float
) -> PredictionRecord:
    db_prediction = PredictionRecord(
        user_id=user_id,
        text=text,
        label=label,
        score=score
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_predictions_by_user(
    db: Session,
    user_id: str,
    skip: int = 0,
    limit: int = 100
) -> List[PredictionRecord]:
    return db.query(PredictionRecord)\
        .filter(PredictionRecord.user_id == user_id)\
        .offset(skip)\
        .limit(limit)\
        .all()

def get_prediction_by_id(
    db: Session,
    prediction_id: int
) -> Optional[PredictionRecord]:
    return db.query(PredictionRecord)\
        .filter(PredictionRecord.id == prediction_id)\
        .first()

def get_all_predictions(
    db: Session,
    skip: int = 0,
    limit: int = 100
) -> List[PredictionRecord]:
    return db.query(PredictionRecord)\
        .offset(skip)\
        .limit(limit)\
        .all()
