from fastapi.testclient import TestClient
from src.api.main import app    # change en src.api.app si besoin

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

def test_recommend():
    response = client.post("/recommend", json={"user_id": 1, "n_recommendations": 5})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 5