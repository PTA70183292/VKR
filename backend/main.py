from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, status  # основные инструменты фастапи для создания апи
from fastapi.security import OAuth2PasswordRequestForm     # специальная форма для авторизации в сваггере
from sqlalchemy.orm import Session                         # тип данных сессии для работы с базой данных
from typing import List, Optional                          # инструменты для подсказки типов данных
import pandas as pd                                        # библиотека пандас для работы с таблицами и csv
import os                                                  # модуль для работы с операционной системой и путями
import shutil                                              # утилиты для копирования и удаления папок и файлов
import json                                                # библиотека для работы с форматом json
import zipfile                                             # модуль для распаковки zip архивов с моделями
from pydantic import BaseModel                             # базовый класс для проверки валидации данных

# импорты модулей нашего проекта
from config import settings                                 # импортируем настройки проекта из файла config
from database import get_db, init_db, SessionLocal          # функции для подключения к нашей базе данных
import schemas                                              # схемы pydantic для валидации входящих запросов
from ml_model import get_sentiment_model, SentimentModel    # функции для загрузки и использования нейросети
from training import SentimentTrainer                       # класс который отвечает за процесс обучения модели
import crud                                                 # функции для создания чтения и обновления записей в бд
import models                                               # наши модели таблиц базы данных
import security                                             # функции безопасности для паролей и токенов

# СИСТЕМА КОНФИГУРАЦИИ
CONFIG_FILE = "system_config.json"                                            # имя файла для хранения глобальных настроек системы (активная модель)

class ActiveModelRequest(BaseModel):                                          # pydantic схема для валидации запроса на смену модели
    model_name: str                                                           # поле с именем модели, которое должно быть строкой

def get_active_model_name():                                                  # функция для получения имени текущей активной модели
    # читает имя активной модели из json файла
    if not os.path.exists(CONFIG_FILE):                                       # проверяем, существует ли файл конфигурации
        return "QLoRA r64"                                                    # если файла нет, возвращаем дефолтное имя модели
    try:                                                                      # начинаем блок попытки чтения файла
        with open(CONFIG_FILE, "r") as f:                                     # открываем файл конфигурации в режиме чтения
            data = json.load(f)                                               # загружаем json данные в словарь
            return data.get("active_model", "QLoRA r64")                      # возвращаем значение active_model или дефолт
    except:                                                                   # если произошла любая ошибка при чтении
        return "QLoRA r64"                                                    # возвращаем дефолтное значение для надежности

def save_active_model_name(name: str):                                        # функция для сохранения выбора активной модели
    # сохраняет выбор админа чтобы не сбросился при рестарте
    with open(CONFIG_FILE, "w") as f:                                         # открываем файл конфигурации в режиме записи
        json.dump({"active_model": name}, f)                                  # записываем словарь с именем модели в json

app = FastAPI(                                                                # инициализируем основное приложение FastAPI
    title=settings.app_title,                                                 # устанавливаем заголовок из настроек
    version=settings.app_version                                              # устанавливаем версию приложения из настроек
)

# ГЛОБАЛЬНОЕ СОСТОЯНИЕ
training_status = {                                                           # словарь для хранения статуса обучения в оперативной памяти
    "is_training": False,                                                     # флаг: идет ли сейчас обучение
    "progress": 0,                                                            # прогресс обучения (в процентах или шагах)
    "status": "idle",                                                         # текущий текстовый статус (ожидание, обучение, ошибка)
    "message": "",                                                            # сообщение для пользователя о текущем этапе
    "history": []                                                             # список для хранения истории прошлых обучений
}
trainer_instance = None                                                       # глобальная переменная для хранения экземпляра класса тренера

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def restore_history_from_disk():                                              # функция для восстановления истории обучения при рестарте
    # сканируем папку и восстанавливаем историю для графиков
    base_path = "./trained_models"                                            # путь к папке, где хранятся обученные модели
    restored_history = []                                                     # пустой список для восстановленной истории

    if not os.path.exists(base_path):                                         # если папки с моделями не существует
        return []                                                             # возвращаем пустой список

    for model_name in os.listdir(base_path):                                  # перебираем все папки внутри trained_models
        model_dir = os.path.join(base_path, model_name)                       # формируем полный путь к конкретной модели
        history_file = os.path.join(model_dir, "training_history.json")       # формируем путь к файлу истории этой модели

        if os.path.isdir(model_dir) and os.path.exists(history_file):         # проверяем, что это папка и внутри есть файл истории
            try:                                                              # попытка прочитать json файл
                with open(history_file, 'r') as f:                            # открываем файл истории на чтение
                    data = json.load(f)                                       # парсим json
                    if isinstance(data, list) and data:                       # проверяем, что данные это список и он не пуст
                        info = data[-1]                                       # берем последнюю запись (финальный результат обучения)
                        info["model_name"] = model_name                       # добавляем имя модели в словарь данных
                        restored_history.append(info)                         # добавляем восстановленную запись в общий список
            except Exception as e:                                            # перехватываем ошибки чтения
                print(f"Ошибка чтения истории {model_name}: {e}")             # выводим ошибку в консоль

    try:                                                                      # попытка отсортировать историю
        restored_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True) # сортируем по дате, новые сверху
    except:                                                                   # если сортировка не удалась
        pass                                                                  # просто пропускаем этот шаг
        
    return restored_history                                                   # возвращаем список восстановленной истории


def get_next_assignee(db: Session) -> str:                                    # функция для выбора исполнителя тикета (алгоритм round-robin)
    # 1. получаем список сотрудников
    agents = db.query(models.User).filter(models.User.role == "user").order_by(models.User.username).all() # запрос к БД: найти всех с ролью user и отсортировать
    agent_usernames = [u.username for u in agents]                            # создаем список только из имен пользователей
    
    # если сотрудников нет кидаем на главного админа из конфига
    if not agent_usernames:                                                   # если список сотрудников пуст
        return settings.superadmin_email                                      # возвращаем email супер-админа как резервный вариант

    # 2. ищем последний созданный тикет
    last_ticket = db.query(models.Ticket).order_by(models.Ticket.created_at.desc()).first() # запрос последнего тикета по дате создания

    # 3. определяем следующего по кругу
    if not last_ticket or not last_ticket.assigned_to or last_ticket.assigned_to not in agent_usernames: # если это первый тикет или прошлый был ничей
        return agent_usernames[0]                                             # назначаем первому сотруднику в списке
    
    try:                                                                      # попытка найти индекс следующего сотрудника
        current_index = agent_usernames.index(last_ticket.assigned_to)        # находим индекс того, кто делал последний тикет
        next_index = (current_index + 1) % len(agent_usernames)               # вычисляем следующий индекс по кругу (остаток от деления)
        return agent_usernames[next_index]                                    # возвращаем имя следующего сотрудника
    except ValueError:                                                        # если предыдущий сотрудник был удален из списка
        return agent_usernames[0]                                             # начинаем распределение с начала списка


def process_ticket_analysis(ticket_id: int):                                  # фоновая задача для анализа текста тикета
    # фоновая задача для анализа текста нейросетью
    db = SessionLocal()                                                       # создаем новую сессию БД специально для этого потока
    try:                                                                      # блок обработки ошибок
        ticket = crud.get_ticket_by_id(db, ticket_id)                         # получаем тикет из базы по ID
        if not ticket: return                                                 # если тикет не найден, выходим из функции

        model = get_sentiment_model()                                         # получаем глобальный экземпляр ML модели (singleton)
        
        target_model = ticket.model_name if ticket.model_name else "QLoRA r64" # определяем имя модели: из тикета или дефолтное
        
        sentiment_result = model.predict(ticket.description, model_name=target_model) # запускаем предсказание тональности
        
        ticket.label = sentiment_result["label"]                              # записываем полученную метку (класс) в тикет
        ticket.score = sentiment_result["score"]                              # записываем уверенность модели в тикет
        
        db.commit()                                                           # сохраняем изменения в базе данных
        print(f" [Background] Тикет #{ticket_id} готов: {ticket.label}")      # логируем успех в консоль
        
    except Exception as e:                                                    # если произошла ошибка при анализе
        print(f" [Background] Ошибка анализа: {e}")                           # выводим текст ошибки в консоль
    finally:                                                                  # блок, который выполняется всегда
        db.close()                                                            # закрываем сессию БД, чтобы не было утечек памяти


@app.on_event("startup")                                                      # декоратор FastAPI: выполнить функцию при запуске сервера
def startup_event():                                                          # функция инициализации
    # действия при запуске сервера
    global training_status                                                    # объявляем, что будем менять глобальную переменную
    
    training_status["history"] = restore_history_from_disk()                  # 1. загружаем историю обучений с диска в память
    init_db()                                                                 # 2. создаем таблицы в базе данных (если их нет)
    get_sentiment_model()                                                     # 3. загружаем ML модель в память (прогрев)
    
    # 4. создаем пользователей из конфига если их нет
    db = next(get_db())                                                       # получаем сессию БД вручную
    try:                                                                      # попытка создать пользователей
        # берем данные из переменных окружения (безопасно)
        users_list = [                                                        # список кортежей с данными пользователей из конфига
            (settings.superadmin_email, "admin"),                           
            (settings.manager_email, "manager"),                            
            (settings.default_user_email_1, "user"),                        
            (settings.default_user_email_2, "user"),                       
            (settings.default_student_email, "student")                       
        ]
        for username, role in users_list:                                     # перебираем список пользователей
            if not crud.get_user_by_username(db, username):                   # если пользователя с таким именем нет в базе
                pwd_hash = security.get_password_hash(username)               # хешируем пароль
                new_user = models.User(username=username, hashed_password=pwd_hash, role=role) # создаем объект пользователя
                db.add(new_user)                                              # добавляем в сессию
        db.commit()                                                           # сохраняем всех пользователей в БД
        print("✅ Пользователи проверены")                                    # логируем успех
    except Exception as e:                                                    # если ошибка при создании пользователей
        print(f"⚠️ Ошибка юзеров: {e}")                                       # выводим ошибку
    finally:                                                                  # в любом случае
        db.close()                                                            # закрываем сессию БД


# АВТОРИЗАЦИЯ
@app.post("/token")                                                           # эндпоинт для получения токена (логин)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)): # параметры: форма логина и сессия БД
    # выдает токен по логину и паролю
    user = crud.get_user_by_username(db, form_data.username)                  # ищем пользователя в базе по имени
    if not user or not security.verify_password(form_data.password, user.hashed_password): # проверяем существование и пароль
        raise HTTPException(status_code=401, detail="Неверный логин или пароль") # если неверно, возвращаем ошибку 401
    
    # создаем токен и вшиваем туда роль
    access_token = security.create_access_token(data={"sub": user.username, "role": user.role}) # генерируем JWT токен
    return {"access_token": access_token, "role": user.role, "username": user.username} # возвращаем токен и данные юзера


# Обращения

@app.post("/tickets", response_model=schemas.TicketResponse)                  # эндпоинт создания тикета, возвращает схему ответа
def create_ticket(                                                            # функция обработки запроса
    ticket_in: schemas.TicketCreate,                                          # валидация входящих данных через Pydantic
    background_tasks: BackgroundTasks,                                        # инструмент для запуска фоновых задач
    db: Session = Depends(get_db),                                            # инъекция сессии базы данных
    current_user: models.User = Depends(security.get_current_user)            # проверка авторизации и получение текущего юзера
):
    # создает тикет и запускает анализ
    creator_email = current_user.username                                     # получаем email создателя из токена
    
    active_model_name = get_active_model_name()                               # узнаем имя текущей активной модели из конфига
    assignee = get_next_assignee(db)                                          # автоматически назначаем исполнителя

    db_ticket = models.Ticket(                                                # создаем объект модели тикета
        user_email=creator_email,                                             # email автора
        subject=ticket_in.subject,                                            # тема обращения
        description=ticket_in.description,                                    # текст обращения
        label="Анализ...",                                                    # временная метка, пока идет анализ
        score=0.0,                                                            # начальная уверенность 0
        model_name=active_model_name,                                         # фиксируем, какой моделью будем проверять
        assigned_to=assignee,                                                 # записываем назначенного исполнителя
        status="Новое"                                                        # начальный статус тикета
    )
    db.add(db_ticket)                                                         # добавляем тикет в сессию
    db.commit()                                                               # сохраняем в базу данных
    db.refresh(db_ticket)                                                     # обновляем объект, чтобы получить присвоенный ID
    
    background_tasks.add_task(process_ticket_analysis, db_ticket.id)          # добавляем задачу анализа текста в фоновую очередь
    return db_ticket                                                          # возвращаем созданный тикет пользователю


@app.get("/tickets", response_model=List[schemas.TicketResponse])             # эндпоинт получения списка тикетов
def get_tickets(                                                              # функция обработки
    db: Session = Depends(get_db),                                            # сессия БД
    current_user: models.User = Depends(security.get_current_user)            # текущий авторизованный пользователь
):
    # возвращает список с учетом прав доступа
    query = db.query(models.Ticket)                                           # создаем базовый запрос ко всем тикетам
    
    if current_user.role == "student":                                        # если пользователь студент
        query = query.filter(models.Ticket.user_email == current_user.username) # фильтруем: только его собственные тикеты
    elif current_user.role == "user":                                         # если пользователь сотрудник
        query = query.filter(models.Ticket.assigned_to == current_user.username) # фильтруем: только назначенные ему задачи
        
    return query.order_by(models.Ticket.created_at.desc()).all()              # сортируем от новых к старым и возвращаем список


@app.get("/tickets/{ticket_id}", response_model=schemas.TicketResponse)       # эндпоинт получения одного тикета по ID
def get_ticket_detail(                                                        # функция обработки
    ticket_id: int,                                                           # ID тикета из URL
    db: Session = Depends(get_db),                                            # сессия БД
    current_user: models.User = Depends(security.get_current_user)            # текущий пользователь
):
    # детали конкретного тикета
    ticket = crud.get_ticket_by_id(db, ticket_id)                             # получаем тикет через CRUD функцию
    if not ticket:                                                            # если тикет не найден
        raise HTTPException(status_code=404, detail="Не найдено")             # возвращаем ошибку 404
        
    # защита чтобы студент не подсмотрел чужое по id
    if current_user.role == "student" and ticket.user_email != current_user.username: # если студент пытается открыть чужой тикет
         raise HTTPException(status_code=403, detail="Это не ваше обращение") # возвращаем ошибку доступа 403
         
    return ticket                                                             # возвращаем данные тикета


@app.put("/tickets/{ticket_id}/label", response_model=schemas.TicketResponse) # эндпоинт для обновления метки
def update_ticket_label_route(                                                # функция обработки
    ticket_id: int,                                                           # ID тикета
    label_data: schemas.TicketLabelUpdate,                                    # данные для обновления (новая метка)
    db: Session = Depends(get_db),                                            # сессия БД
    current_user: models.User = Depends(security.get_current_user)            # текущий пользователь
):
    # ручная правка метки человеком human in the loop
    ticket = crud.get_ticket_by_id(db, ticket_id)                             # ищем тикет
    if not ticket:                                                            # если не нашли
        raise HTTPException(status_code=404, detail="Не найдено")             # ошибка 404
        
    updated_ticket = crud.update_ticket_label(db, ticket_id, label_data.label) # обновляем метку через CRUD функцию
    return updated_ticket                                                     # возвращаем обновленный тикет
# --- CONFIG ENDPOINTS (УПРАВЛЕНИЕ КОНФИГУРАЦИЕЙ) ---
@app.get("/config/active-model")                                              # эндпоинт для получения текущей модели
def get_active_model_endpoint(                                                # функция обработки
    current_user: models.User = Depends(security.get_current_user)            # проверяем авторизацию
):
    return {"model_name": get_active_model_name()}                            # возвращаем имя из json конфига

@app.post("/config/active-model")                                             # эндпоинт для смены активной модели
def set_active_model_endpoint(                                                # функция обработки
    req: ActiveModelRequest,                                                  # ожидаем json с именем модели
    current_user: models.User = Depends(security.get_current_user)            # проверяем авторизацию
):
    if current_user.role != "admin":                                          # только админ может менять глобальную модель
        raise HTTPException(status_code=403, detail="Только админ может менять глобальную модель")
    
    if req.model_name != "QLoRA r64":                                         # если это не дефолтная модель
        model_path = os.path.join("./trained_models", req.model_name)         # строим путь к папке
        if not os.path.exists(model_path):                                    # проверяем наличие папки на диске
             raise HTTPException(status_code=404, detail="Модель не найдена на сервере")
             
    save_active_model_name(req.model_name)                                    # сохраняем выбор в файл
    return {"message": "Active model updated", "model_name": req.model_name}  # возвращаем успех


# --- УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ ---
@app.get("/users", response_model=List[schemas.UserResponse])                 # эндпоинт получения списка пользователей
def get_users(current_user: models.User = Depends(security.get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin":                                          # защита: только админ видит список
        raise HTTPException(status_code=403, detail="Доступ запрещен")
    return db.query(models.User).all()                                        # возвращаем всех из таблицы users

@app.post("/users", response_model=schemas.UserResponse)                      # эндпоинт создания пользователя вручную
def create_new_user(user_in: schemas.UserCreate, current_user: models.User = Depends(security.get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin": raise HTTPException(403, "Forbidden")    # защита доступа
    if crud.get_user_by_username(db, user_in.username): raise HTTPException(400, "User exists") # проверка на дубликаты
    return crud.create_user(db, user_in, user_in.role)                        # создаем через crud

@app.delete("/users/username/{username}")                                     # эндпоинт удаления пользователя
def delete_user(username: str, current_user: models.User = Depends(security.get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin": raise HTTPException(403, "Forbidden")    # защита доступа
    user = crud.get_user_by_username(db, username)                            # ищем пользователя
    if not user: raise HTTPException(404, "Not found")                        # если не найден
    
    # используем настройку из config.py для защиты суперадмина
    if user.username == settings.superadmin_email: raise HTTPException(400, "Нельзя удалить администратора")
    
    db.delete(user)                                                           # удаляем запись
    db.commit()                                                               # фиксируем изменения
    return {"message": "Deleted"}                                             # возвращаем успех


# ЗАГРУЗКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
@app.post("/training/upload-dataset")                                         # эндпоинт загрузки csv файла
async def upload_dataset(                                                     # асинхронная функция (важно для файлов)
    file: UploadFile = File(...),                                             # получаем файл из формы
    current_user: models.User = Depends(security.get_current_user)            # требуем авторизацию
):
    # 2. Проверяем роль
    if current_user.role != "admin":                                          # только админ может грузить данные
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Только администратор может загружать датасеты"
        )
    
    if not file.filename.endswith('.csv'):                                    # простая валидация расширения
        raise HTTPException(status_code=400, detail="Только CSV файлы поддерживаются")
    
    os.makedirs("./datasets", exist_ok=True)                                  # создаем папку если нет
    file_path = f"./datasets/{file.filename}"                                 # формируем путь сохранения
    
    contents = await file.read()                                              # читаем байты файла
    with open(file_path, "wb") as f:                                          # сохраняем на диск
        f.write(contents)
    
    try:
        # список кодировок для перебора (excel часто сохраняет в cp1251)
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
        df = None
        used_encoding = None
        last_error = None

        for enc in encodings_to_try:                                          # пробуем открыть файл разными кодировками
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=enc,
                    sep=",",                                                  # разделитель запятая
                    engine="python",
                    quotechar='"',
                    skip_blank_lines=True,
                    on_bad_lines="skip"                                       # пропускаем битые строки
                )
                used_encoding = enc
                break                                                         # если получилось - выходим из цикла

            except Exception as e:
                last_error = e

        if df is None:                                                        # если ни одна кодировка не подошла
            raise ValueError(f"Не удалось прочитать CSV: {last_error}")

        # Чистим Excel-мусор (пустые колонки Unnamed)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.dropna(axis=1, how="all")                                     # удаляем колонки где все значения пустые

        required_columns = ["text", "label"]                                  # обязательные поля
        missing = [c for c in required_columns if c not in df.columns]        # проверяем их наличие

        if missing:                                                           # если чего-то нет - ошибка
            raise HTTPException(
                status_code=400,
                detail=f"CSV должен содержать колонки {required_columns}. Найдено: {list(df.columns)}"
            )

        return {                                                              # возвращаем статистику по файлу
            "filename": file.filename,
            "path": file_path,            
            "rows": len(df),
            "columns": list(df.columns),
            "label_distribution": df["label"].value_counts().to_dict()        # распределение классов (сколько позитива/негатива)
        }

    except Exception as e:                                                    # если ошибка при обработке
        if os.path.exists(file_path):
            os.remove(file_path)                                              # удаляем битый файл
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")


@app.get("/training/models-list")                                             # эндпоинт списка моделей
def get_trained_models_list():
    # Возвращает список доступных обученных моделей
    base_path = "./trained_models"
    if not os.path.exists(base_path):
        return {"models": []}
    
    try:
        models = [
            name for name in os.listdir(base_path)                            # читаем имена папок
            if os.path.isdir(os.path.join(base_path, name)) 
            and not name.startswith("_temp")                                  # фильтруем временные папки обучения
        ]
        return {"models": models}
    except Exception:
        return {"models": []}


@app.post("/training/upload-model-zip")                                       # эндпоинт загрузки готовой модели (zip)
async def upload_model_zip_endpoint(
    file: UploadFile = File(...),
    current_user: models.User = Depends(security.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Только администратор может загружать модели")

    if not file.filename.endswith(".zip"):                                    # проверяем расширение
        raise HTTPException(status_code=400, detail="Файл должен быть архивом .zip")

    # 1. Определяем целевое имя папки из имени файла
    model_name = file.filename.rsplit(".", 1)[0]
    target_dir = os.path.join("./trained_models", model_name)

    if os.path.exists(target_dir):                                            # нельзя перезаписать существующую модель
        raise HTTPException(status_code=400, detail=f"Модель '{model_name}' уже существует.")

    # 2. Временные пути
    temp_zip_path = f"temp_{file.filename}"
    temp_extract_dir = f"temp_extract_{model_name}"

    try:
       
        with open(temp_zip_path, "wb") as f:                                   # Сохраняем ZIP
            content = await file.read()
            f.write(content)

    
        os.makedirs(temp_extract_dir, exist_ok=True)                           # Распаковываем во временную папку
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:                   # Проверка на безопасность путей (Zip Slip уязвимость)
            for name in zip_ref.namelist():
                if ".." in name or name.startswith("/"):
                    raise HTTPException(status_code=400, detail="Небезопасный архив")
            zip_ref.extractall(temp_extract_dir)

      
        
        extracted_items = os.listdir(temp_extract_dir)                         # Проверяем содержимое распакованной папки (часто папка внутри папки)
        
        source_dir = temp_extract_dir
        if len(extracted_items) == 1:                                         # Если там всего 1 папка (эффект матрешки), заходим внутрь
            potential_sub = os.path.join(temp_extract_dir, extracted_items[0])
            if os.path.isdir(potential_sub):
                source_dir = potential_sub                                    # Меняем источник на вложенную папку

        # Перемещаем файлы в итоговую папку models
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

        return {"message": f"Модель '{model_name}' успешно загружена", "path": target_dir}

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Файл поврежден")
    except Exception as e:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)                     # удаляем недокачанное
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    finally:
        # Чистим мусор (временные архивы и папки)
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)


# ЗАПУСК ОБУЧЕНИЯ (ФОНОВАЯ ЗАДАЧА)
def run_training_task(                                                        # функция которая выполняется в фоне
    dataset_path: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    custom_model_name: str,
    source_model_path: Optional[str] = None
):
    global training_status, trainer_instance                                  # доступ к глобальному состоянию

    training_status["is_training"] = True                                     # ставим флаг "занято"

    try:
        training_status["status"] = "loading_dataset"
        training_status["message"] = "Загрузка датасета..."

        trainer_instance = SentimentTrainer()                                 # создаем экземпляр тренера
        dataset = trainer_instance.load_dataset_from_csv(dataset_path)        # грузим данные

        training_status["status"] = "preparing_data"
        training_status["message"] = "Подготовка данных..."

        train_dataset, eval_dataset = trainer_instance.prepare_dataset(dataset) # делим на train/test

        training_status["status"] = "setting_up_model"
        if source_model_path:
             training_status["message"] = f"Загрузка весов из {source_model_path}..."
        else:
             training_status["message"] = "Инициализация базовой модели..."

        trainer_instance.setup_model_for_training(source_model_path=source_model_path) # готовим нейросеть

        training_status["status"] = "training"
        training_status["message"] = f"Обучение модели '{custom_model_name}' ({num_epochs} эпох)..."

        # Временная папка для чекпоинтов обучения
        temp_output = f"./trained_models/_temp_{custom_model_name}"

        # ЗАПУСК ТРЕНИРОВКИ (самый долгий процесс)
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

        # Сохраняем финальную версию
        model_path = trainer_instance.save_model(custom_name=custom_model_name)

        # Удаляем временные файлы чекпоинтов (экономим место)
        shutil.rmtree(temp_output, ignore_errors=True)

        training_status["status"] = "completed"
        training_status["message"] = "Обучение завершено успешно"
        # Добавляем запись в историю
        training_status["history"].append({
            **training_info,
            "model_path": model_path,
            "model_name": custom_model_name
        })

    except Exception as e:
        import traceback
        traceback.print_exc()                                                 # печатаем полный трейс ошибки в консоль

        training_status["status"] = "error"
        training_status["message"] = f"Ошибка обучения: {str(e)}"             # сообщаем об ошибке на фронт

    finally:
        training_status["is_training"] = False                                # снимаем флаг "занято"


@app.post("/training/start")                                                  # эндпоинт кнопки "Начать обучение"
async def start_training(
    background_tasks: BackgroundTasks,                                        # инструмент для запуска фона
    body: schemas.TrainingStartRequest,                                       # параметры обучения из json
    current_user: models.User = Depends(security.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Обучение доступно только администратору")
    
    global training_status

    if training_status["is_training"]:                                        # если уже что-то учится - отказ
        raise HTTPException(status_code=400, detail="Обучение уже выполняется")

    if not os.path.exists(body.dataset_path):                                 # проверка наличия файла
        raise HTTPException(
            status_code=404,
            detail=f"Датасет не найден: {body.dataset_path}"
        )

    
    background_tasks.add_task(                                                # Добавляем тяжелую задачу в очередь (ответ пользователю уйдет сразу)
        run_training_task, 
        body.dataset_path,
        body.num_epochs,
        body.batch_size,
        body.learning_rate,
        body.custom_model_name,
        body.source_model_path
    )

    return {                                                                  # быстрый ответ
        "message": "Обучение запущено",
        "dataset_path": body.dataset_path,
        "model_name": body.custom_model_name,
        "parameters": {
            "num_epochs": body.num_epochs,
            "batch_size": body.batch_size,
            "learning_rate": body.learning_rate
        }
    }


@app.get("/training/status", response_model=schemas.TrainingStatusResponse)   # эндпоинт для прогресс-бара
def get_training_status(
    current_user: models.User = Depends(security.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещен")
        
    global training_status
    return training_status                                                    # возвращаем текущее состояние


@app.get("/training/history")                                                 # эндпоинт для графиков
def get_training_history(
    current_user: models.User = Depends(security.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещен")
        
    global training_status
    return {"history": training_status["history"]}                            # возвращаем массив с историей


@app.post("/training/load-model")                                             # эндпоинт для проверки загрузки модели
def load_trained_model(
    model_path: str,
    current_user: models.User = Depends(security.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещен")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Модель не найдена")
    
    try:
        global trainer_instance
        if trainer_instance is None:
            trainer_instance = SentimentTrainer()                             # инициализируем тренера если нет
        trainer_instance.load_trained_model(model_path)                       # пробуем загрузить веса в память
        
        return {"message": "Модель загружена успешно", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")


@app.get("/health")                                                           # эндпоинт проверки жизни сервера
def health_check():
    return {"status": "ok"}                                                   # используется докером healthcheck


@app.post("/training/reset")                                                  # эндпоинт аварийного сброса
def reset_training_status(
    current_user: models.User = Depends(security.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Доступ запрещен")

    global training_status
    training_status = {                                                       # Сбрасываем статус на "свободен", но сохраняем историю
        "is_training": False,
        "progress": 0,
        "status": "idle",
        "message": "",
        "history": training_status.get("history", []) 
    }
    return {"message": "Training status reset"}

if __name__ == "__main__":                                                    # точка входа
    import uvicorn                                                            # сервер uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)                               # запуск приложения на порту 8000
