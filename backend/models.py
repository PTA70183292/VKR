from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from datetime import datetime
from database import Base  # Импортируем Base, созданный в database.py

# МОДЕЛЬ ПОЛЬЗОВАТЕЛЯ (Сотрудники и Администраторы)
class User(Base):
    __tablename__ = "users"  # Имя таблицы в PostgreSQL

    # Первичный ключ, индексируется для скорости поиска
    id = Column(Integer, primary_key=True, index=True)
    
    # Логин (email), должен быть уникальным
    username = Column(String, unique=True, index=True, nullable=False)
    
    # Храним ТОЛЬКО хеш пароля
    hashed_password = Column(String, nullable=False)
    
    # Роль: 'admin', 'manager', 'user' (или 'student')
    role = Column(String, default="user") 
    
    # Флаг активности. 
    is_active = Column(Boolean, default=True)

# МОДЕЛЬ ОБРАЩЕНИЯ (ТИКЕТА)
class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Автор обращения. Храним email как строку
    user_email = Column(String, index=True, nullable=False)
    
    subject = Column(String, nullable=False)                # Тема обращения
    description = Column(Text, nullable=False)              # Полный текст (тип Text не имеет лимита длины)
    
    # Блок ML-анализа
    # Результат работы нейросети (LABEL_0, LABEL_1...)
    label = Column(String, nullable=False)
    # Уверенность модели (например, 0.98)
    score = Column(Float, nullable=False)
    # Имя модели, которая приняла решение. По умолчанию "QLoRA r64"
    model_name = Column(String, default="QLoRA r64")
    
    # Блок управления процессом
    status = Column(String, default="Новое")                # Статусы: Новое -> В работе -> Закрыто
    assigned_to = Column(String, nullable=True, index=True) # Кто обрабатывает (логин сотрудника)
    
    # Время создания (UTC).
    created_at = Column(DateTime, default=datetime.utcnow)
