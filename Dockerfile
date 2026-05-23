# Dockerfile для развертывания модели
FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование модели и API
COPY catboost_risk_model.pkl .
COPY model_metadata.json .
COPY risk_api.py .

# Открываем порт
EXPOSE 5000

# Запуск API
CMD ["python", "risk_api.py"]
