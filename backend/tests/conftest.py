import pytest
import sys
import os

# добавляем пути чтобы тесты видели папки проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from database import Base, get_db
from main import app
from security import create_access_token
import models # импортируем модели чтобы создать юзеров

TEST_DATABASE_URL = "sqlite:///:memory:" # база в памяти для скорости

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def test_db():
    Base.metadata.create_all(bind=engine) # создаем таблицы
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine) # чистим после теста

@pytest.fixture(scope="function")
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db # подменяем реальную бд на тестовую
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

# ФИКСТУРЫ ДЛЯ ЮЗЕРОВ (создаем их в базе перед тестом)

@pytest.fixture
def admin_headers(test_db):
    user = models.User(username="admin_test", hashed_password="fake", role="admin")
    test_db.add(user) # сохраняем админа в базу
    test_db.commit()
    token = create_access_token(data={"sub": "admin_test", "role": "admin"})
    return {"Authorization": f"Bearer {token}"} # возвращаем готовый заголовок

@pytest.fixture
def student_headers(test_db):
    user = models.User(username="student_test", hashed_password="fake", role="student")
    test_db.add(user) # сохраняем студента в базу
    test_db.commit()
    token = create_access_token(data={"sub": "student_test", "role": "student"})
    return {"Authorization": f"Bearer {token}"}
