FROM python:3.12-slim
WORKDIR /app
# Системні утиліти (компілятор для пакетів без готових wheel)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# Python-залежності із зафіксованими версіями
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Код та дані експерименту
COPY src/ src/
COPY data/raw/ data/raw/
COPY train.py .
ENV PYTHONHASHSEED=42
CMD ["python", "train.py"]
