from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import io
import os

from config import settings
from database import get_db, init_db
from schemas import PredictRequest, PredictResponse, TrainingRequest, TrainingStatusResponse
from ml_model import get_sentiment_model, SentimentModel
from training import SentimentTrainer
import crud

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version
)


training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "",
    "history": []
}

trainer_instance = None

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

@app.post("/training/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы поддерживаются")
    
    os.makedirs("./datasets", exist_ok=True)
    file_path = f"./datasets/{file.filename}"
    
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    try:
        df = pd.read_csv(file_path)
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"CSV должен содержать колонки: {required_columns}"
            )
        
        return {
            "filename": file.filename,
            "path": file_path,
            "rows": len(df),
            "columns": list(df.columns),
            "label_distribution": df['label'].value_counts().to_dict()
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")

def run_training_task(
    dataset_path: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float
):
    global training_status, trainer_instance
    
    try:
        training_status["is_training"] = True
        training_status["status"] = "loading_dataset"
        training_status["message"] = "Загрузка датасета..."
        
        trainer_instance = SentimentTrainer()
        dataset = trainer_instance.load_dataset_from_csv(dataset_path)
        
        training_status["status"] = "preparing_data"
        training_status["message"] = "Подготовка данных..."
        
        train_dataset, eval_dataset = trainer_instance.prepare_dataset(dataset)
        
        training_status["status"] = "setting_up_model"
        training_status["message"] = "Настройка модели..."
        
        trainer_instance.setup_model_for_training()
        
        training_status["status"] = "training"
        training_status["message"] = f"Обучение модели ({num_epochs} эпох)..."
        
        training_info = trainer_instance.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        training_status["status"] = "saving"
        training_status["message"] = "Сохранение модели..."
        
        model_path = trainer_instance.save_model()
        
        training_status["status"] = "completed"
        training_status["message"] = "Обучение завершено успешно"
        training_status["history"].append({
            **training_info,
            "model_path": model_path
        })
        
    except Exception as e:
        training_status["status"] = "error"
        training_status["message"] = f"Ошибка обучения: {str(e)}"
    finally:
        training_status["is_training"] = False

@app.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    dataset_path: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4
):
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Обучение уже выполняется")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Датасет не найден")
    
    background_tasks.add_task(
        run_training_task,
        dataset_path,
        num_epochs,
        batch_size,
        learning_rate
    )
    
    return {
        "message": "Обучение запущено",
        "dataset_path": dataset_path,
        "parameters": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    }

@app.get("/training/status", response_model=TrainingStatusResponse)
def get_training_status():

    return training_status

@app.get("/training/history")
def get_training_history():

    return {"history": training_status["history"]}

@app.post("/training/load-model")
def load_trained_model(model_path: str):

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Модель не найдена")
    
    try:
        global trainer_instance
        trainer_instance = SentimentTrainer()
        trainer_instance.load_trained_model(model_path)
        
        return {
            "message": "Модель загружена успешно",
            "model_path": model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")

@app.get("/health")
def health_check():

    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
