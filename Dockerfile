# Используем официальный образ Python с меньшим весом
FROM python:3.12-slim

# Устанавливаем зависимости ОС
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное
COPY . .

# Указываем порт (Streamlit по умолчанию 8501)
EXPOSE 8501

# Команда запуска
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
