from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Конфигурация приложения.
    Pydantic автоматически считывает переменные из файла .env,
    сопоставляя имена полей (case-insensitive) с ключами в файле.
    """
    # Переменные должны совпадать с .env (postgres_user -> POSTGRES_USER)
    postgres_user: str
    postgres_password: str
    postgres_db: str
    
    # Мы берем готовую строку подключения прямо из .env
    database_url: str 

    base_model_name: str
    adapter_name: str

    app_title: str
    app_version: str

    secret_key: str
    algorithm: str
    access_token_expire_minutes: int

    superadmin_email: str
    manager_email: str 

    default_user_email_1: str 
    default_user_email_2: str 
    default_student_email: str

    # Технические настройки Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env",            # Имя файла
        env_file_encoding="utf-8",  # Кодировка
        extra="ignore"              # Игнорировать лишние переменные в .env, если они есть
    )

# Инициализация настроек
settings = Settings()
