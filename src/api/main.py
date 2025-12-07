# =======================================
# FastAPI — SVD Recommendation API
# =======================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

app = FastAPI(
    title="Système de Recommandation de Films – MLOps 2025 (SVD)",
    description="""**Groupe** : Ranim Ben Mrad • Iheb Melki • Hiba Zaibi<br>
                   **Enseignant** : Sonia Gharsalli<br>
                   **Dataset** : MovieLens 100K<br>
                   **Modèle utilisé** : SVD Collaborative Filtering""",
    version="2.0.0"
)

# ---------------------------------------
# Load data & trained SVD model
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

movies = pd.read_csv(BASE_DIR / "data/processed/movies_processed.csv.dvc")
ratings = pd.read_csv(BASE_DIR / "data/processed/ratings_processed.csv.dvc")

model_path = BASE_DIR / "models" / "svd_model.pkl"
if not model_path.exists():
    raise RuntimeError(f"❌ SVD model not found at {model_path}")

with open(model_path, "rb") as f:
    svd_model = pickle.load(f)

user_ids = svd_model["user_ids"]
movie_ids = svd_model["movie_ids"]
user_factors = svd_model["user_factors"]
movie_factors = svd_model["movie_factors"]

# Create fast lookup
user_to_index = {u: i for i, u in enumerate(user_ids)}
movie_to_index = {m: i for i, m in enumerate(movie_ids)}


# ---------------------------------------
# Request Body
# ---------------------------------------
class Request(BaseModel):
    user_id: int = Field(..., example=42, description="ID utilisateur (1–610)")
    n_recommendations: int = Field(10, ge=1, le=20, example=10)


# ---------------------------------------
# Health Check
# ---------------------------------------
@app.get("/", tags=["Health"])
def home():
    return {"message": "API SVD en ligne", "status": "OK"}


# ---------------------------------------
# Recommendation Endpoint
# ---------------------------------------
@app.post("/recommend", tags=["Recommandation"])
def recommend(req: Request):

    # 1️⃣ Validate user
    if req.user_id not in user_to_index:
        raise HTTPException(404, f"Utilisateur {req.user_id} inconnu dans le modèle SVD")

    u_idx = user_to_index[req.user_id]

    # 2️⃣ Compute predicted scores for all movies
    scores = np.dot(user_factors[u_idx], movie_factors.T)

    scores_series = pd.Series(scores, index=movie_ids)

    # 3️⃣ Remove movies already rated by the user
    user_rated_movies = ratings[ratings["userId"] == req.user_id]["movieId"].tolist()
    scores_series = scores_series.drop(labels=user_rated_movies, errors="ignore")

    # 4️⃣ top-K recommendations
    top_movies = scores_series.nlargest(req.n_recommendations)

    # 5️⃣ Build response
    results = []
    for movie_id, score in top_movies.items():
        title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
        results.append({
            "movieId": int(movie_id),
            "title": title,
            "score": round(float(score), 3)
        })

    return {
        "user_id": req.user_id,
        "recommendations": results
    }
