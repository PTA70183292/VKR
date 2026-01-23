from pydantic import BaseModel           # Базовый класс для всех схем валидации
from datetime import datetime            # Для работы с датой и временем
from typing import Optional, Dict, List  # Типы данных для аннотаций

# Авторизация
class UserCreate(BaseModel):
    # Схема для регистрации пользователя (входящие данные)
    username: str
    password: str
    role: str = "user"  # По умолчанию создаем обычного пользователя

class UserResponse(BaseModel):
    # Схема для ответа с данными пользователя (исходящие данные). Пароль здесь не возвращаем!
    id: int
    username: str
    role: str
    is_active: bool
    
    class Config:
        # В Pydantic V2 это позволяет создавать схему напрямую из объекта SQLAlchemy (ORM)
        from_attributes = True

class Token(BaseModel):
    # Схема для выдачи JWT токена.
    access_token: str
    token_type: str
    role: str
    username: str

# Обращения
class TicketCreate(BaseModel):
    # Схема создания тикета (от фронтенда)
    subject: str
    description: str 
    user_email: str

class TicketResponse(BaseModel):
    # Схема для отображения тикета в списке.
    id: int
    user_email: str
    subject: str
    description: str  
    label: str
    score: float
    model_name: str
    status: str
    
    # Optional означает, что поле может быть None (если исполнитель еще не назначен)
    assigned_to: Optional[str] = None 
    
    created_at: datetime
    
    class Config:
        from_attributes = True  # Разрешаем чтение данных из ORM-модели

# Обучение
class TrainingStatusResponse(BaseModel):
    # Схема статуса обучения (прогресс-бар)
    is_training: bool
    progress: int
    status: str
    message: str
    history: List[Dict]  # Список словарей с историей loss/accuracy

class TrainingStartRequest(BaseModel):
    # Параметры запуска обучения (гиперпараметры)
    dataset_path: str
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    custom_model_name: str = "my_model"
    source_model_path: Optional[str] = None # Если None, используется базовая модель

class TicketLabelUpdate(BaseModel):
    # Схема для ручного изменения тональности (Human-in-the-loop)
    label: str
