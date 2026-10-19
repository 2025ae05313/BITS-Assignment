"""Render docs/architecture.png (run: python docs/make_architecture.py)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).with_name("architecture.png")
C = {"data": "#DCEAF7", "ml": "#E3F1DF", "ci": "#FCE9D6", "k8s": "#E8E1F3", "mon": "#FBE0E0",
     "edge": "#3A3A3A"}

fig, ax = plt.subplots(figsize=(15, 8.6))
ax.set_xlim(0, 150)
ax.set_ylim(0, 86)
ax.axis("off")


def group(x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec="#9A9A9A", lw=1, ls="--", alpha=0.55))
    ax.text(x + 1.2, y + h - 2.2, title, fontsize=11, fontweight="bold", color="#333", va="top")


def box(x, y, w, h, text, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1",
                                fc=fc, ec=C["edge"], lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, linespacing=1.35)
    return (x, y, w, h)


def arrow(a, b, label=None, side="r", style="-|>", color="#444", rad=0.0, lx=0, ly=0):
    ax_, ay, aw, ah = a
    bx, by, bw, bh = b
    if side == "r":
        p1, p2 = (ax_ + aw, ay + ah / 2), (bx, by + bh / 2)
    elif side == "d":
        p1, p2 = (ax_ + aw / 2, ay), (bx + bw / 2, by + bh)
    elif side == "u":
        p1, p2 = (ax_ + aw / 2, ay + ah), (bx + bw / 2, by)
    else:
        p1, p2 = (ax_, ay + ah / 2), (bx + bw, by + bh / 2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, color=color,
                                 lw=1.3, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label, fontsize=7.6,
                ha="center", color="#333", style="italic",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))


ax.text(75, 84, "Heart Disease Risk Prediction — MLOps Architecture", ha="center",
        fontsize=15, fontweight="bold")

# ---- data & experimentation ----
group(1, 44, 50, 36, "1  Data & experimentation", C["data"])
uci = box(4, 66, 19, 8, "UCI ML Repository\nprocessed.cleveland.data\n(GitHub mirror fallback)")
dl = box(28, 66, 20, 8, "download_data.py\nvalidate · clean\nbinary target")
eda = box(4, 49, 19, 10, "EDA (notebook 01)\nhistograms · heatmap\nclass balance")
trn = box(28, 49, 20, 10, "train.py (notebook 02)\nColumnTransformer +\nLogReg / RandomForest\n5-fold CV grid search")
arrow(uci, dl, "HTTP", ly=3)
arrow(dl, eda, side="d", rad=0.0, label="data/processed", lx=-6)
arrow(dl, trn, side="d")

# ---- tracking & registry ----
group(55, 44, 40, 36, "2  Tracking & packaging", C["ml"])
mlf = box(58, 63, 34, 11, "MLflow tracking (SQLite + ./mlruns)\n68 runs: params · CV & test metrics\nROC/PR/confusion/importance plots")
pkg = box(58, 48, 16, 11, "models/\nmodel.joblib\nmodel_metadata.json\nmlflow_model/")
reg = box(77, 48, 15, 11, "Model registry\nheart-disease-\nclassifier\n@champion")
arrow(trn, mlf, "log runs", ly=2)
arrow(mlf, reg, side="d", label="register", lx=5)
arrow(trn, pkg, "best model", ly=-3)

# ---- CI/CD ----
group(99, 44, 50, 36, "3  CI/CD — GitHub Actions", C["ci"])
gh = box(102, 66, 20, 8, "git push / PR\nto GitHub")
lint = box(126, 66, 20, 8, "lint\nflake8")
test = box(126, 55, 20, 8, "test\npytest · 34 tests\ncoverage ≥ 80 %")
tr2 = box(102, 55, 20, 8, "train\nretrain · quality gate\n(CV AUC ≥ 0.85)")
dock = box(102, 46.5, 44, 6, "docker: build → smoke-test /predict → push GHCR")
arrow(gh, lint)
arrow(lint, test, side="d")
arrow(test, tr2, side="l")
arrow(tr2, dock, side="d")
ax.text(124, 44.6, "artifacts: lint report · junit + coverage · model · mlruns · image · logs",
        fontsize=7.4, ha="center", color="#555")

# ---- Kubernetes ----
group(1, 2, 98, 38, "4  Serving — Kubernetes (Docker Desktop)", C["k8s"])
img = box(52, 27, 28, 9.5, "Docker image\nheart-disease-api:1.0.0\nslim · multi-stage · non-root")
pods = box(48, 5, 48, 17, "Deployment  heart-api  (2 replicas, rolling update)\n\n"
           "FastAPI + Uvicorn\nPOST /predict  ·  /predict/batch\nGET /health  /ready  /model  /metrics\n"
           "readiness + liveness probes · resource limits")
svc = box(25, 9, 18, 10, "Service\ntype: LoadBalancer\nlocalhost:8080\n(+ optional Ingress)")
client = box(4, 9, 16, 10, "Client\ncurl · /docs\nload_test.py")
arrow(pkg, img, side="d", label="COPY model", lx=-6)
arrow(img, pods, side="d", label="kubectl apply -k", lx=9)
arrow(client, svc, "JSON", ly=2.5)
arrow(svc, pods)

# ---- monitoring ----
group(102, 2, 47, 38, "5  Monitoring & logging", C["mon"])
prom = box(105, 25, 19, 9, "Prometheus\npod discovery\nalert rules")
graf = box(128, 25, 18, 9, "Grafana\nprovisioned\ndashboard")
logs = box(105, 8, 41, 10, "Structured JSON logs (stdout)\none line per request + per prediction\nrequest_id · status · latency · P(disease)\nkubectl logs / any log aggregator")
arrow(pods, prom, "scrape /metrics", ly=3)
arrow(prom, graf)
arrow(pods, logs, rad=0.15, label="stdout", lx=0, ly=-3)

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
