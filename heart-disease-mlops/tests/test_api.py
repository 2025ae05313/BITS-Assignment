"""API tests using FastAPI's TestClient against a small fixture model."""
import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.schemas import PatientFeatures


@pytest.fixture
def client(trained_model_dir, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", trained_model_dir / "model.joblib")
    monkeypatch.setattr(api, "METADATA_PATH", trained_model_dir / "model_metadata.json")
    with TestClient(api.app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_version"] == "test-0001"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=PatientFeatures.example())
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    expected = body["probability_disease"] if body["prediction"] else 1 - body["probability_disease"]
    assert body["confidence"] == pytest.approx(expected, abs=1e-4)
    assert body["risk_band"] in ("low", "moderate", "high")
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_accepts_missing_optional_fields(client):
    payload = PatientFeatures.example()
    payload.pop("ca")
    payload["thal"] = None
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [("age", 150), ("cp", 9), ("chol", -1),
                                         ("sex", "male")])
def test_predict_rejects_invalid_input(client, field, value):
    payload = PatientFeatures.example() | {field: value}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_unknown_and_missing_fields(client):
    assert client.post("/predict", json=PatientFeatures.example() | {"bmi": 30}).status_code == 422
    payload = PatientFeatures.example()
    payload.pop("age")
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"instances": [PatientFeatures.example()] * 3})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_metrics_endpoint_counts_predictions(client):
    client.post("/predict", json=PatientFeatures.example())
    text = client.get("/metrics").text
    assert "model_predictions_total" in text
    assert 'api_requests_total{endpoint="/predict"' in text
    assert "api_request_latency_seconds_bucket" in text


def test_model_card(client):
    r = client.get("/model")
    assert r.status_code == 200 and r.json()["model_family"] == "logistic_regression"


def test_service_unavailable_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api.ModelStore, "pipeline", None)
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=PatientFeatures.example()).status_code == 503
