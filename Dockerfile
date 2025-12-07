FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 🔥 Ajout pour que Python trouve src/
ENV PYTHONPATH="/app"

CMD ["python", "src/training/train.py"]
