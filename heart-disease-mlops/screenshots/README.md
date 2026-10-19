# Screenshot checklist

✅ = already captured from a real run (MLflow UI, pytest, local API). ⬜ = capture on your machine;
the report has a matching placeholder box — replace it with the image.


| | # | File | How to capture |
|---|---|---|---|
| ✅ | 1 | `01_mlflow_experiments.png` | `make mlflow-ui` → experiment *heart-disease-classification*, runs table (parent + nested runs visible) |
| ✅ | 2 | `02_mlflow_compare.png` | select the `logistic_regression` and `random_forest` runs → **Compare** → metrics chart |
| ✅ | 3 | `03_mlflow_lr_run.png`, `03b_mlflow_artifacts.png` | open the `logistic_regression` run → Artifacts → `plots/roc_curve.png` |
| ✅ | 4 | `04_mlflow_registry.png` | Models tab → `heart-disease-classifier` with alias `champion` |
| ⬜ | 5 | `05_github_actions_run.png` | GitHub → Actions → a green run showing lint → test → train → docker |
| ⬜ | 6 | `06_github_actions_summary.png` | the run's Summary page (test + model tables, artifacts list) |
| ✅ | 7 | `07_pytest.png` | terminal: `pytest -v` (34 passed) |
| ⬜ | 8 | `08_docker_build_run.png` | terminal: `docker build` finishing + `docker run` + curl `/predict` response |
| ⬜ | 9 | `09_k8s_resources.png` | terminal: `kubectl -n heart-api get deploy,pods,svc` (2/2 Running, LoadBalancer localhost:8080) |
| ⬜ | 10 | `10_k8s_predict.png` | terminal: curl `http://localhost:8080/predict` with response |
| ⬜ | 11 | `11_swagger.png` | browser: `http://localhost:8080/docs` with a `/predict` call executed |
| ⬜ | 12 | `12_grafana_dashboard.png` | Grafana dashboard after `scripts/load_test.py` |
| ⬜ | 13 | `13_prometheus_targets.png` | Prometheus → Status → Targets (2 pods UP) |
| ⬜ | 14 | `14_api_logs.png` | terminal: `kubectl -n heart-api logs -l app=heart-api --tail=20` |

Tip: run `python scripts/load_test.py --url http://localhost:8080 --n 300` before
captures 12–14 so the dashboard and logs have data.

Also captured: `08a_api_local_smoke_test.png` (API run directly with uvicorn — the Docker version, #8, still needs your capture).
