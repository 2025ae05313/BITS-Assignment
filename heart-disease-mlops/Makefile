IMAGE ?= heart-disease-api:1.0.0
API   ?= http://localhost:8080

.PHONY: help setup data eda train test lint mlflow-ui serve docker-build docker-run \
        k8s-deploy k8s-status monitoring load-test k8s-clean all

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:           ## install all dependencies into the active venv
	pip install -r requirements.txt && pip install -e .

data:            ## download + clean the UCI dataset
	python scripts/download_data.py

eda:             ## regenerate EDA figures
	python scripts/run_eda.py

train:           ## tune, track (MLflow) and package the model
	python scripts/train.py

test:            ## unit + integration tests with coverage
	pytest -v --cov=src/heartml --cov=app --cov-report=term-missing

lint:            ## flake8
	flake8 src app tests scripts

mlflow-ui:       ## open MLflow UI on http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:           ## run the API without Docker on :8000
	uvicorn app.main:app --reload --port 8000

docker-build:    ## build the serving image
	docker build -t $(IMAGE) .

docker-run:      ## run the image on :8000 and send a sample request
	docker run -d --rm --name heart-api -p 8000:8000 $(IMAGE)
	@sleep 4
	curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json; echo

k8s-deploy:      ## deploy to the current kube context (Docker Desktop)
	kubectl apply -k k8s/
	kubectl -n heart-api rollout status deploy/heart-api --timeout=120s

k8s-status:      ## show deployed resources and hit the endpoint
	kubectl -n heart-api get deploy,pods,svc -o wide
	curl -s -X POST $(API)/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json; echo

monitoring:      ## deploy Prometheus + Grafana to Kubernetes
	kubectl apply -k monitoring/
	kubectl -n monitoring rollout status deploy/grafana --timeout=180s

load-test:       ## generate traffic for the dashboard
	python scripts/load_test.py --url $(API) --n 300

k8s-clean:       ## remove everything from the cluster
	kubectl delete -k monitoring/ --ignore-not-found
	kubectl delete -k k8s/ --ignore-not-found

all: data eda train test lint  ## full local pipeline
