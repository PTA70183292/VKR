from sqlalchemy import create_engine  # Функция для создания соединения с БД
from sqlalchemy.ext.declarative import declarative_base  # Базовый класс для создания ORM-моделей
from sqlalchemy.orm import sessionmaker  # Фабрика для создания сессий (транзакций)
from config import settings  # Импортируем наши настройки (где лежит database_url)

# Создаем "движок" базы данных.
# settings.database_url подтягивается из config.py (а он берет из .env)
engine = create_engine(settings.database_url)

# Создаем фабрику сессий.
# autocommit=False: мы сами управляем транзакциями (commit/rollback), это надежнее.
# autoflush=False: данные не отправляются в БД автоматически без явной команды.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс, от которого будут наследоваться все наши таблицы (User, Ticket и т.д.)
Base = declarative_base()


# Зависимость (Dependency) для FastAPI.Создает новую сессию для каждого запроса, а после завершения — закрывает её.

def get_db():
    db = SessionLocal()  # Открываем сессию
    try:
        yield db  # Передаем сессию в функцию-обработчик (endpoint)
    finally:
        db.close()  # Гарантированно закрываем сессию (возвращаем подключение в пул)


# Cоздает таблицы в базе данных, если их нет. Обычно вызывается при старте приложения в main.py.
def init_db():
    Base.metadata.create_all(bind=engine)
