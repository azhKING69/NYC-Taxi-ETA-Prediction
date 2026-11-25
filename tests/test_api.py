import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from serving.app import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_predict_endpoint_valid():
    with TestClient(app) as client:
        payload = {
            "PULocationID": 161,
            "DOLocationID": 237,
            "trip_distance": 3.2,
            "pickup_datetime": "2026-05-15T08:30:00",
            "passenger_count": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_duration_minutes" in data
        assert data["predicted_duration_minutes"] > 0
        assert data["status"] == "success"


def test_predict_endpoint_invalid_zone():
    with TestClient(app) as client:
        payload = {
            "PULocationID": 999,  # Invalid: must be <= 265
            "DOLocationID": 237,
            "trip_distance": 3.2,
            "pickup_datetime": "2026-05-15T08:30:00",
            "passenger_count": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error


def test_dashboard_endpoint():
    with TestClient(app) as client:
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "NYC Taxi MLOps Command Center" in response.text
