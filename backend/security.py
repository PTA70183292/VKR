from datetime import datetime, timedelta, timezone# Классы для работы со временем (срок действия токена)
from typing import Optional                       # Типизация (необязательные аргументы)
from jose import JWTError, jwt                    # Библиотека для кодирования/декодирования JWT
from passlib.context import CryptContext          # Безопасное хеширование паролей
from fastapi.security import OAuth2PasswordBearer # Схема авторизации OAuth2 (Bearer token)
from fastapi import Depends, HTTPException, status # Утилиты FastAPI для зависимостей и ошибок
from sqlalchemy.orm import Session                # Тип сессии базы данных
from database import get_db                       # Функция получения сессии БД
import models                                     # Импорт моделей БД (User)
from config import settings                       # Настройки (SECRET_KEY, ALGORITHM)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") # Настройка контекста хеширования (bcrypt - стандарт индустрии)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Схема OAuth2: сообщает Swagger UI, куда отправлять логин/пароль

def verify_password(plain_password, hashed_password):  #Сравнивает 'чистый' пароль с хешем из БД.
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):                       # Превращает пароль в хеш (для регистрации/смены пароля)
    return pwd_context.hash(password)

def create_access_token(data: dict):                   # Генерация JWT токена
    to_encode = data.copy()
    
    # Вычисляем время истечения (текущее UTC + минуты из конфига)
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})             # Добавляем метку времени 'exp' в payload
    
    # Кодируем словарь в строку JWT, используя SECRET_KEY и алгоритм
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)): #Зависимость: проверяет токен и возвращает пользователя
    
    # Заготовка ошибки 401 Unauthorized (если токен невалиден)
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm]) # Расшифровка токена ключом из настроек
        username: str = payload.get("sub")        # Извлекаем имя пользователя
        
        if username is None:                      # Если имени нет в токене — ошибка
            raise credentials_exception
            
    except JWTError:                              # Ошибка подписи или истек срок действия
        raise credentials_exception
        

    user = db.query(models.User).filter(models.User.username == username).first()  # Ищем пользователя в БД по имени из токена
    
    if user is None:                              # Если пользователь удален, но токен жив
        raise credentials_exception
        
    return user                                   # Возвращаем объект пользователя