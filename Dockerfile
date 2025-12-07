FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/api/main.py .

# Download data at startup (MovieLens 100K – 10 Mo seulement)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget -O data.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip && \
    unzip data.zip -d data && \
    mv data/ml-latest-small data/raw && \
    rm data.zip

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
