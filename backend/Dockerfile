FROM python:3.10-slim

WORKDIR /app

# 1. Устанавливаем системные зависимости (git и build-essential часто нужны для ML-библиотек)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Обновляем pip до последней версии (критично для torch/bitsandbytes)
RUN pip install --upgrade pip

COPY requirements.txt .

# 3. Устанавливаем зависимости с увеличенным тайм-аутом (чтобы torch не обрывался)
# Флаг --no-cache-dir экономит место в образе
RUN pip install --default-timeout=1000 -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
