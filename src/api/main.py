from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

app = FastAPI(
    title="Système de Recommandation de Films – MLOps 2025",
    description="""**Groupe** : Ranim Ben Mrad • Iheb Melki • Hiba Zaibi<br>
                   **Enseignant** : Sonia Gharsalli<br>
                   **Dataset** : MovieLens 100K<br>
                   **État** : Semaine 5** : 100 % terminé""",
    version="1.0.0"
)

BASE_DIR = Path(__file__).parent.parent.parent
movies = pd.read_csv(BASE_DIR / "data/raw/movies.csv")
ratings = pd.read_csv(BASE_DIR / "data/raw/ratings.csv")
matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

# Modèle simple mais efficace (toujours dispo même sans MLflow)
model = NearestNeighbors(n_neighbors=40, metric="cosine")
model.fit(matrix.values)

class Request(BaseModel):
    user_id: int = Field(..., example=42, description="ID de l'utilisateur (1 à 610)")
    n_recommendations: int = Field(10, ge=1, le=20, example=10)

@app.get("/", tags=["Santé"])
def home():
    return {"message": "API Recommandation Films – Groupe Ranim • Iheb • Hiba", "status": "ONLINE"}

@app.post("/recommend", tags=["Recommandation"])
def recommend(req: Request):
    if req.user_id not in matrix.index:
        raise HTTPException(404, f"Utilisateur {req.user_id} inconnu")
    
    user_vec = matrix.loc[req.user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vec, n_neighbors=req.n_recommendations + 30)
    similar = matrix.index[indices.flatten()[1:req.n_recommendations + 30]]
    pred = matrix.loc[similar].mean(axis=0).nlargest(req.n_recommendations)

    result = []
    for mid in pred.index:
        title = movies[movies.movieId == mid].title.values[0]
        result.append({"movieId": int(mid), "title": title, "score": round(float(pred[mid]), 3)})
    
    return {"user_id": req.user_id, "recommendations": result}
