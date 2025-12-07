# Dockerfile
FROM python:3.11-slim

# On installe git uniquement pour calmer MLflow (sinon warning rouge dans les logs)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie requirements en premier pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code
COPY . .

# Variables utiles
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

# Render attend que tu écoutes sur la variable $PORT (par défaut 10000)
EXPOSE 10000

# Commande de démarrage : uvicorn sur le port fourni par Render
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-10000}