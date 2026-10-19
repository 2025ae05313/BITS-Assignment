# What's left before submitting

Everything in the repo is built and verified. These steps need your machine / accounts.

## 1. Push to GitHub (≈5 min)

```bash
cd heart-disease-mlops
git init && git add . && git commit -m "Heart disease MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/heart-disease-mlops.git
git push -u origin main
```

The push triggers the **ci-cd** workflow (Actions tab). Wait for all four jobs to go green,
then capture screenshots 05 and 06. To publish the image to GHCR the repo's
Settings → Actions → General → Workflow permissions must allow "Read and write".

## 2. Docker + Kubernetes on Docker Desktop (≈10 min)

Docker Desktop → Settings → Kubernetes → Enable Kubernetes.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

docker build -t heart-disease-api:1.0.0 .                       # screenshot 08
docker run -d --rm --name heart-api -p 8000:8000 heart-disease-api:1.0.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_request.json
docker stop heart-api

kubectl config use-context docker-desktop
kubectl apply -k k8s/
kubectl -n heart-api rollout status deploy/heart-api
kubectl -n heart-api get deploy,pods,svc                        # screenshot 09
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d @scripts/sample_request.json   # 10
# browser: http://localhost:8080/docs                           # screenshot 11

kubectl apply -k monitoring/
kubectl -n monitoring rollout status deploy/grafana
python scripts/load_test.py --url http://localhost:8080 --n 300
# Grafana http://localhost:3000 (admin/admin)                   # screenshot 12
# Prometheus http://localhost:9090 → Status → Targets           # screenshot 13
kubectl -n heart-api logs -l app=heart-api --tail=20            # screenshot 14
```

Windows PowerShell: replace `-d @scripts/sample_request.json` with
`Invoke-RestMethod -Method Post -Uri http://localhost:8080/predict -ContentType 'application/json' -InFile scripts/sample_request.json`.

If port 8080 is taken, change `port: 8080` in `k8s/service.yaml`.

## 3. Finish the report

Open `docs/MLOps_Assignment1_Report.docx`:
- title page: BITS ID, repository URL, video link
- replace each dashed **[Insert screenshot …]** box with the matching image from `screenshots/`
- §12 Links: repository URL
- export to PDF if required (File → Save As → PDF)

## 4. Demo video (3–5 min) — suggested script

1. **(30 s)** Repo tour: `src/heartml`, `app/`, `tests/`, `k8s/`, `monitoring/`, `.github/workflows/ci.yml`.
2. **(45 s)** `python scripts/download_data.py` and `python scripts/train.py` running; open MLflow UI
   (`make mlflow-ui`): nested runs, the selected run's metrics and plots, registry `@champion`.
3. **(45 s)** GitHub Actions: a green run, open the summary (test + model tables) and artifacts.
4. **(45 s)** `docker build` → `kubectl apply -k k8s/` → `kubectl get pods,svc` → curl `/predict`
   (one high-risk and one low-risk patient) → invalid request returning 422 → Swagger `/docs`.
5. **(45 s)** `load_test.py` running → Grafana dashboard filling in → Prometheus targets →
   `kubectl logs` showing JSON request lines.
6. **(15 s)** Close on the architecture diagram.

## 5. Clean up

```bash
make k8s-clean
```
