"""FastAPI model-serving app.

Endpoints
  GET  /           service info
  GET  /health     liveness  (process is up)
  GET  /ready      readiness (model loaded)
  GET  /model      model card (metrics, params, version)
  POST /predict    single patient -> prediction + confidence
  POST /predict/batch
  GET  /metrics    Prometheus metrics

Every request is logged as one JSON line (request id, path, status, latency) and
every prediction is logged with its outcome, so logs can be shipped to any
aggregator; the same signals are exported as Prometheus metrics.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.schemas import (
    BatchRequest,
    BatchResponse,
    PatientFeatures,
    PredictionResponse,
)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
METADATA_PATH = Path(
    os.environ.get("MODEL_METADATA_PATH", MODEL_PATH.parent / "model_metadata.json")
)
FEATURE_ORDER = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca",
                 "sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


# ---------------------------------------------------------------- logging ----
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("heart_api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


logger = _build_logger()


def log_event(msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


# ---------------------------------------------------------------- metrics ----
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
PROBABILITY = Histogram("model_prediction_probability", "P(disease) distribution",
                        buckets=[i / 10 for i in range(1, 11)])
ERRORS = Counter("api_errors_total", "Unhandled errors", ["endpoint"])
MODEL_INFO = Gauge("model_info", "Loaded model (value always 1)", ["version", "family"])
IN_FLIGHT = Gauge("api_requests_in_progress", "Requests currently being served")


# ------------------------------------------------------------------ model ----
class ModelStore:
    pipeline = None
    metadata: dict = {}

    @classmethod
    def load(cls) -> None:
        cls.pipeline = joblib.load(MODEL_PATH)
        cls.metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        MODEL_INFO.labels(cls.version(), cls.metadata.get("model_family", "unknown")).set(1)
        log_event("model_loaded", path=str(MODEL_PATH), version=cls.version())

    @classmethod
    def version(cls) -> str:
        return str(cls.metadata.get("model_version", "unknown"))

    @classmethod
    def threshold(cls) -> float:
        return float(cls.metadata.get("decision_threshold", 0.5))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ModelStore.load()
    except Exception as err:  # keep serving /health so the failure is visible
        log_event("model_load_failed", error=str(err), path=str(MODEL_PATH))
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description="Predicts heart-disease risk from 13 clinical attributes (UCI Cleveland).",
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    endpoint = request.url.path
    start = time.perf_counter()
    IN_FLIGHT.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    except Exception:
        ERRORS.labels(endpoint).inc()
        raise
    finally:
        elapsed = time.perf_counter() - start
        IN_FLIGHT.dec()
        if endpoint != "/metrics":  # don't let scrapes drown the logs
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            log_event("request", request_id=request_id, method=request.method,
                      path=endpoint, status=status, latency_ms=round(elapsed * 1000, 2),
                      client=request.client.host if request.client else None)


def _risk_band(p: float) -> str:
    return "low" if p < 0.3 else "moderate" if p < 0.6 else "high"


def _predict(rows: list[PatientFeatures], request_id: str) -> list[PredictionResponse]:
    if ModelStore.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    frame = pd.DataFrame([r.model_dump() for r in rows])[FEATURE_ORDER].astype(float)
    probs = ModelStore.pipeline.predict_proba(frame)[:, 1]
    out = []
    for i, p in enumerate(probs):
        p = float(p)
        pred = int(p >= ModelStore.threshold())
        label = "heart_disease" if pred else "no_heart_disease"
        PREDICTIONS.labels(label).inc()
        PROBABILITY.observe(p)
        rid = request_id if len(probs) == 1 else f"{request_id}-{i}"
        log_event("prediction", request_id=rid, prediction=pred,
                  probability_disease=round(p, 4), model_version=ModelStore.version())
        out.append(PredictionResponse(
            prediction=pred, label=label,
            confidence=round(p if pred else 1 - p, 4),
            probability_disease=round(p, 4), risk_band=_risk_band(p),
            model_version=ModelStore.version(), request_id=rid))
    return out


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs",
            "model_version": ModelStore.version()}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if ModelStore.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": ModelStore.version()}


@app.get("/model")
def model_card():
    if not ModelStore.metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    keys = ("model_name", "model_version", "model_family", "best_params", "cv_metrics",
            "test_metrics", "decision_threshold", "features", "trained_at_utc")
    return {k: ModelStore.metadata.get(k) for k in keys}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    return _predict([patient], request.state.request_id)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    return BatchResponse(predictions=_predict(batch.instances, request.state.request_id))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
