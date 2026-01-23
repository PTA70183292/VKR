from sqlalchemy.orm import Session      # Тип данных для сессии БД
from models import Ticket, User         # Импортируем наши ORM-модели (таблицы)
from security import get_password_hash  # Функция хеширования (разберем в следующем шаге)

# ПОЛЬЗОВАТЕЛИ
def get_user_by_username(db: Session, username: str):
    # Ищет пользователя в БД по логину (для авторизации)
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user, role: str = "user"):
    # Создает нового пользователя
    # 1. Хешируем пароль
    hashed_password = get_password_hash(user.password)
    
    # 2. Создаем объект модели User
    db_user = User(
        username=user.username, 
        hashed_password=hashed_password, 
        role=role
    )
    
    # 3. Добавляем в сессию и сохраняем (commit)
    db.add(db_user)
    db.commit()
    
    # 4. Обновляем объект (чтобы получить присвоенный ID)
    db.refresh(db_user)
    return db_user

# ТИКЕТЫ
def create_ticket(
    db: Session,
    user_email: str,
    subject: str,
    description: str,  # Сюда придет содержимое (description из схемы)
    label: str,
    score: float,
    model_name: str
) -> Ticket:
    # Создает запись обращения в БД с результатами ML-анализа.
    
    db_ticket = Ticket(
        user_email=user_email,
        subject=subject,
        # Слева - имя колонки в БД, Справа - переменная с данными
        description=description,  
        label=label,
        score=score,
        model_name=model_name,
        status="Новое"  # Статус по умолчанию
    )
    
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    return db_ticket

def get_all_tickets(db: Session, limit: int = 100):
    #Возвращает последние тикеты (сортировка от новых к старым)
    return db.query(Ticket).order_by(Ticket.created_at.desc()).limit(limit).all()

def get_ticket_by_id(db: Session, ticket_id: int):
    # Получает один тикет по ID (для детального просмотра)
    return db.query(Ticket).filter(Ticket.id == ticket_id).first()

def update_ticket_label(db: Session, ticket_id: int, new_label: str):
    
    # Обновляет метку тональности вручную.
    # Используется, когда сотрудник исправляет ошибку ИИ.
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        ticket.label = new_label
        ticket.model_name = "Manual"  # Маркируем, что это ручная правка
        ticket.score = 1.0            # Уверенность 100%, так как проверил человек
        
        db.commit()      # Сохраняем изменения
        db.refresh(ticket) # Обновляем данные в объекте
    return ticket
