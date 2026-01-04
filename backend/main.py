from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import io
import os
import shutil 

from config import settings
from database import get_db, init_db
import json
from schemas import PredictRequest, PredictResponse, TrainingStatusResponse, TrainingStartRequest
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
def restore_history_from_disk():
    """Сканирует папки моделей и собирает историю обучения"""
    base_path = "./trained_models"
    restored_history = []

    if not os.path.exists(base_path):
        return []

    # Проходим по всем папкам в trained_models
    for model_name in os.listdir(base_path):
        model_dir = os.path.join(base_path, model_name)
        history_file = os.path.join(model_dir, "training_history.json")

        if os.path.isdir(model_dir) and os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # data обычно это список, берем последний элемент или все
                    if isinstance(data, list) and data:
                        # Берем последнюю запись обучения
                        info = data[-1]
                        # Добавляем имя модели, если его нет внутри JSON
                        info["model_name"] = model_name
                        restored_history.append(info)
            except Exception as e:
                print(f"Error reading history for {model_name}: {e}")

    # Сортируем по дате (если есть timestamp), от новых к старым
    try:
        restored_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    except:
        pass
        
    return restored_history

@app.on_event("startup")
def startup_event():
    init_db()
    get_sentiment_model()

    print("Восстановление истории обучения...")
    history = restore_history_from_disk()
    training_status["history"] = history
    print(f"Восстановлено {len(history)} логов обучения.")

@app.get("/training/models-list")
def get_trained_models_list():
    """Возвращает список доступных обученных моделей"""
    base_path = "./trained_models"
    if not os.path.exists(base_path):
        return {"models": []}
    
    try:
        models = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        # Сортируем, чтобы новые были сверху (опционально)
        models.sort(reverse=True)
        return {"models": models}
    except Exception:
        return {"models": []}
    
@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
    model: SentimentModel = Depends(get_sentiment_model)
):
    # ПЕРЕДАЕМ model_name В ПРЕДСКАЗАНИЕ
    result = model.predict(req.text, model_name=req.model_name)
    
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
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
        df = None
        used_encoding = None
        last_error = None

        for enc in encodings_to_try:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=enc,
                    sep=",",
                    engine="python",
                    quotechar='"',
                    skip_blank_lines=True,
                    on_bad_lines="skip"
                )
                used_encoding = enc
                break

            except Exception as e:
                last_error = e

        if df is None:
            raise ValueError(f"Не удалось прочитать CSV: {last_error}")

        # 🧹 Чистим Excel-мусор
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.dropna(axis=1, how="all")

        required_columns = ["text", "label"]
        missing = [c for c in required_columns if c not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"CSV должен содержать колонки {required_columns}. Найдено: {list(df.columns)}"
            )

        return {
            "filename": file.filename,
            "path": file_path,            
            "rows": len(df),
            "columns": list(df.columns),
            "label_distribution": df["label"].value_counts().to_dict()
        }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")


@app.get("/training/models-list")
def get_trained_models_list():
    """Возвращает список доступных обученных моделей"""
    base_path = "./trained_models"
    if not os.path.exists(base_path):
        return {"models": []}
    
    # Сканируем папки
    try:
        models = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        return {"models": models}
    except Exception:
        return {"models": []}


def run_training_task(
    dataset_path: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    custom_model_name: str,
    source_model_path: Optional[str] = None
):
    global training_status, trainer_instance

    training_status["is_training"] = True

    try:
        training_status["status"] = "loading_dataset"
        training_status["message"] = "Загрузка датасета..."

        trainer_instance = SentimentTrainer()
        dataset = trainer_instance.load_dataset_from_csv(dataset_path)

        training_status["status"] = "preparing_data"
        training_status["message"] = "Подготовка данных..."

        train_dataset, eval_dataset = trainer_instance.prepare_dataset(dataset)

        training_status["status"] = "setting_up_model"
        if source_model_path:
             training_status["message"] = f"Загрузка весов из {source_model_path}..."
        else:
             training_status["message"] = "Инициализация базовой модели..."

        trainer_instance.setup_model_for_training(source_model_path=source_model_path)

        training_status["status"] = "training"
        training_status["message"] = f"Обучение модели '{custom_model_name}' ({num_epochs} эпох)..."

        # Временная папка для обучения
        temp_output = f"./trained_models/_temp_{custom_model_name}"

        training_info = trainer_instance.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=temp_output,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        training_status["status"] = "saving"
        training_status["message"] = "Сохранение модели..."

        model_path = trainer_instance.save_model(custom_name=custom_model_name)

        # Удаляем временные файлы
        shutil.rmtree(temp_output, ignore_errors=True)

        training_status["status"] = "completed"
        training_status["message"] = "Обучение завершено успешно"
        training_status["history"].append({
            **training_info,
            "model_path": model_path,
            "model_name": custom_model_name
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        training_status["status"] = "error"
        training_status["message"] = f"Ошибка обучения: {str(e)}"

    finally:
        training_status["is_training"] = False

@app.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    body: TrainingStartRequest
):
    global training_status

    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Обучение уже выполняется")

    if not os.path.exists(body.dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Датасет не найден: {body.dataset_path}"
        )

    background_tasks.add_task(
        run_training_task,
        body.dataset_path,
        body.num_epochs,
        body.batch_size,
        body.learning_rate,
        body.custom_model_name,
        body.source_model_path
    )

    return {
        "message": "Обучение запущено",
        "dataset_path": body.dataset_path,
        "model_name": body.custom_model_name,
        "parameters": {
            "num_epochs": body.num_epochs,
            "batch_size": body.batch_size,
            "learning_rate": body.learning_rate
        }
    }

@app.get("/training/status", response_model=TrainingStatusResponse)
def get_training_status():
    global training_status
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

@app.post("/training/reset")
def reset_training_status():
    global training_status
    training_status = {
        "is_training": False,
        "progress": 0,
        "status": "idle",
        "message": "",
        "history": []
    }
    return {"message": "Training status reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)