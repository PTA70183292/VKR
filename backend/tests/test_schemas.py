import pytest
from schemas import TicketCreate

class TestTicketCreateSchema:
    def test_valid_ticket(self):
        # добавляем почту 
        req = TicketCreate(
            subject="Вопрос", 
            description="Как дела?", 
            user_email="test@test.ru" 
        )
        assert req.subject == "Вопрос"
