import pytest
from fastapi import status

class TestHealthCheck:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200 # проверяем что сервер жив

class TestTickets:
    def test_create_ticket_auth(self, client, student_headers):
        # добавляем user_email потому что схема TicketCreate его требует
        payload = {
            "subject": "Проблема с входом",
            "description": "Не могу зайти в ЛК",
            "user_email": "student_test" # скармливаем это схеме чтобы не было ошибки
        }
        response = client.post("/tickets", json=payload, headers=student_headers)
        assert response.status_code == 200 # теперь должно быть 200 вместо 401
        assert response.json()["subject"] == "Проблема с входом"

    def test_admin_see_all(self, client, admin_headers):
        response = client.get("/tickets", headers=admin_headers)
        assert response.status_code == 200 # админ должен иметь доступ

class TestSecurity:
    def test_admin_route_forbidden_for_student(self, client, student_headers):
        # Проверяем, что студент получит 403 (Forbidden), если полезет в список юзеров
        response = client.get("/users", headers=student_headers)
        assert response.status_code == 403 
        assert response.json()["detail"] == "Доступ запрещен"

    def test_unauthorized_access(self, client):
        # Проверяем, что без токена вообще нельзя получить список тикетов
        response = client.get("/tickets")
        assert response.status_code == 401 # Unauthorized

    def test_delete_user_is_protected(self, client, student_headers):
        # Проверяем, что студент не может никого удалить
        response = client.delete("/users/username/admin_test", headers=student_headers)
        assert response.status_code == 403
