# =======================================
# FastAPI — SVD Recommendation API + MONITORING
# =======================================

from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
import json
from datetime import datetime
from collections import defaultdict
import threading
import time

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
from fastapi.responses import Response

# --- App ---
app = FastAPI(title="Movie Recommendation SVD + Monitoring", version="3.0")

BASE_DIR = Path(__file__).resolve().parents[2]

movies = pd.read_csv(BASE_DIR / "data/processed/movies_processed.csv")
ratings = pd.read_csv(BASE_DIR / "data/processed/ratings_processed.csv")

model_path = BASE_DIR / "models" / "svd_model.pkl"
if not model_path.exists():
    raise RuntimeError("Model not found")

with open(model_path, "rb") as f:
    svd_model = pickle.load(f)

user_ids = svd_model["user_ids"]
movie_ids = svd_model["movie_ids"]
user_factors = svd_model["user_factors"]
movie_factors = svd_model["movie_factors"]

user_to_index = {u: i for i, u in enumerate(user_ids)}

# --- Monitoring simple ---
request_counter = 0
lock = threading.Lock()

# --- Prometheus ---
registry = CollectorRegistry()
total_requests = Counter('total_requests', 'Total requests', registry=registry)
duration_hist = Histogram('request_duration_seconds', 'Duration', registry=registry)

@app.middleware("http")
async def metrics_middleware(request: FastAPIRequest, call_next):
    start = time.time()
    response = await call_next(request)
    duration_hist.observe(time.time() - start)
    total_requests.inc()
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type="text/plain")

@app.get("/")
def home():
    return {"status": "OK", "monitoring": "active"}

# --- Evidently drift ---
from src.monitoring.dashboard import router as monitoring_router
app.include_router(monitoring_router)

# --- Request model ---
class RecReq(BaseModel):
    user_id: int
    n_recommendations: int = 10

# --- Recommend ---
@app.post("/recommend")
def recommend(req: RecReq):
    global request_counter
    start = datetime.now()

    if req.user_id not in user_to_index:
        raise HTTPException(404, "User not found")

    u_idx = user_to_index[req.user_id]
    scores = np.dot(user_factors[u_idx], movie_factors.T)
    scores_series = pd.Series(scores, index=movie_ids)
    already_rated = ratings[ratings["userId"] == req.user_id]["movieId"].tolist()
    scores_series = scores_series.drop(already_rated, errors="ignore")
    top = scores_series.nlargest(req.n_recommendations)

    results = []
    for mid, score in top.items():
        title = movies.loc[movies["movieId"] == mid, "title"].iloc[0]
        results.append({"movieId": int(mid), "title": title, "score": round(float(score),3)})

    # Log pour Evidently
    from src.monitoring.dashboard import add_request_to_history
    add_request_to_history(req.user_id, req.n_recommendations)

    with lock:
        request_counter += 1

    return {"user_id": req.user_id, "recommendations": results}
