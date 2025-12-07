FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/main.py .

# Télécharge MovieLens 100K au démarrage (10 Mo seulement)
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*
RUN wget -O ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip && \
    unzip ml-latest-small.zip && \
    mv ml-latest-small data && \
    rm ml-latest-small.zip

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
