#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest не установлен. Установка...${NC}"
    pip install pytest pytest-asyncio httpx pytest-cov
fi

mkdir -p tests

if [ -f "test_api.py" ]; then
    mv test_*.py tests/ 2>/dev/null
    mv conftest.py tests/ 2>/dev/null
fi

pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Все тесты пройдены успешно!${NC}"
else
    echo -e "\n${RED}✗ Некоторые тесты не прошли${NC}"
    exit 1
fi

echo "Очистка тестовых баз данных..."
rm -f test*.db

echo -e "\n${GREEN}Готово!${NC}"
