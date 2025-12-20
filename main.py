from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from config import settings
from database import get_db, init_db
from schemas import PredictRequest, PredictResponse
from ml_model import get_sentiment_model, SentimentModel
import crud

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version
)

@app.on_event("startup")
def startup_event():
    init_db()
    get_sentiment_model()

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
    model: SentimentModel = Depends(get_sentiment_model)
):
    result = model.predict(req.text)
    
    # Сохраняем в базу данных
    db_prediction = crud.create_prediction(
        db=db,
        user_id=req.user_id,
        text=req.text,
        label=result["label"],
        score=result["score"]
    )
    
    return db_prediction

@app.get("/predictions/user/{user_id}", response_model=List[PredictResponse])
def get_user_predictions(
    user_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    predictions = crud.get_predictions_by_user(
        db=db,
        user_id=user_id,
        skip=skip,
        limit=limit
    )
    return predictions

@app.get("/predictions/{prediction_id}", response_model=PredictResponse)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    prediction = crud.get_prediction_by_id(db=db, prediction_id=prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@app.get("/predictions", response_model=List[PredictResponse])
def get_all_predictions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    predictions = crud.get_all_predictions(db=db, skip=skip, limit=limit)
    return predictions

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
