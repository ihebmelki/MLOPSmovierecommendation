# =======================================
# FastAPI — SVD Recommendation API
# =======================================

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging, json, threading, time
from collections import defaultdict
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
from fastapi.responses import Response
from datetime import datetime

# ------------------------------
# App FastAPI
# ------------------------------
app = FastAPI(
    title="Système de Recommandation de Films – MLOps 2025 (SVD)",
    description="""**Groupe** : Ranim Ben Mrad • Iheb Melki • Hiba Zaibi<br>
                   **Enseignant** : Mme Sonia Gharsalli<br>
                   **Dataset** : MovieLens 100K<br>
                   **Modèle utilisé** : SVD Collaborative Filtering""",
    version="2.0.0"
)

# ------------------------------
# Load data & model
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

movies = pd.read_csv(BASE_DIR / "data/processed/movies_processed.csv")
ratings = pd.read_csv(BASE_DIR / "data/processed/ratings_processed.csv")

model_path = BASE_DIR / "models" / "svd_model.pkl"
if not model_path.exists():
    raise RuntimeError(f"❌ SVD model not found at {model_path}")

with open(model_path, "rb") as f:
    svd_model = pickle.load(f)

user_ids = svd_model["user_ids"]
movie_ids = svd_model["movie_ids"]
user_factors = svd_model["user_factors"]
movie_factors = svd_model["movie_factors"]

user_to_index = {u: i for i, u in enumerate(user_ids)}
movie_to_index = {m: i for i, m in enumerate(movie_ids)}

# ------------------------------
# Request Body
# ------------------------------
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., example=42, description="ID utilisateur (1–610)")
    n_recommendations: int = Field(10, ge=1, le=20, example=10)

# ------------------------------
# Monitoring global
# ------------------------------
request_counter = 0
user_requests = defaultdict(int)
recommendation_sizes = []

lock = threading.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "user_id": getattr(record, "user_id", None),
            "n_recommendations": getattr(record, "n_recommendations", None),
            "response_time_ms": getattr(record, "response_time_ms", None),
            "message": record.getMessage()
        }
        return json.dumps(log_object)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.handlers = [handler]

# ------------------------------
# Prometheus metrics
# ------------------------------
registry = CollectorRegistry()
request_total = Counter('total_requests', 'Nombre total de requêtes', registry=registry)
request_duration = Histogram('request_duration_seconds', 'Durée des requêtes', registry=registry)

@app.middleware("http")
async def prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    request_total.inc()
    request_duration.observe(duration)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type="text/plain")

# ------------------------------
# Health check
# ------------------------------
@app.get("/", tags=["Health"])
def home():
    return {"message": "API SVD en ligne", "status": "OK"}

@app.get("/health", tags=["Health"])
def health():
    with lock:
        avg_rec = sum(recommendation_sizes[-100:])/len(recommendation_sizes[-100:]) if recommendation_sizes else 10
        top_user = max(user_requests.items(), key=lambda x: x[1])[0] if user_requests else None
        drift_alert = any(count > request_counter*0.5 for count in user_requests.values())

    return {
        "status": "healthy",
        "total_requests": request_counter,
        "top_user_id": top_user,
        "avg_recommendations": round(avg_rec,2),
        "drift_detected": drift_alert,
        "timestamp": datetime.utcnow().isoformat()
    }

# ------------------------------
# Recommendation endpoint
# ------------------------------
@app.post("/recommend", tags=["Recommandation"])
def recommend(req: RecommendationRequest):
    start_time = datetime.now()

    if req.user_id not in user_to_index:
        raise HTTPException(404, f"Utilisateur {req.user_id} inconnu dans le modèle SVD")

    u_idx = user_to_index[req.user_id]
    scores = np.dot(user_factors[u_idx], movie_factors.T)
    scores_series = pd.Series(scores, index=movie_ids)
    user_rated_movies = ratings[ratings["userId"] == req.user_id]["movieId"].tolist()
    scores_series = scores_series.drop(labels=user_rated_movies, errors="ignore")
    top_movies = scores_series.nlargest(req.n_recommendations)

    results = []
    for movie_id, score in top_movies.items():
        title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
        results.append({"movieId": int(movie_id), "title": title, "score": round(float(score),3)})

    duration_ms = (datetime.now() - start_time).total_seconds()*1000
    with lock:
        global request_counter
        request_counter += 1
        user_requests[req.user_id] += 1
        recommendation_sizes.append(req.n_recommendations)

    logger.info("Recommendation served",
                extra={"user_id": req.user_id,
                       "n_recommendations": req.n_recommendations,
                       "response_time_ms": round(duration_ms,2)})

    return {
        "user_id": req.user_id,
        "recommendations": results,
        "meta": {"request_id": request_counter, "served_at": datetime.utcnow().isoformat()}
    }
