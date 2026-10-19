# ---------------------------------------------------------------------------
# Heart Disease Risk API - production image
#   build:  docker build -t heart-disease-api:1.0.0 .
#   run:    docker run -p 8000:8000 heart-disease-api:1.0.0
# Requires a trained model in models/ (run `make train` or download the CI artifact).
# ---------------------------------------------------------------------------

# ---- stage 1: build wheels in an isolated layer ---------------------------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-serve.txt

# ---- stage 2: slim runtime -------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="FastAPI service predicting heart-disease risk (UCI Cleveland)" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_METADATA_PATH=/app/models/model_metadata.json \
    LOG_LEVEL=INFO \
    PORT=8000 \
    WORKERS=1

# non-root user with a fixed numeric UID (lets Kubernetes verify runAsNonRoot)
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home /app app
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY app/ ./app/
COPY models/model.joblib models/model_metadata.json ./models/
RUN chown -R app:app /app
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"

# uvicorn's own access log is disabled: the app writes structured JSON request logs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --no-access-log"]
