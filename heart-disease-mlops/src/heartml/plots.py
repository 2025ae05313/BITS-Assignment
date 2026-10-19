"""Evaluation plots logged to MLflow for every model run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def confusion_matrix_plot(y_true, y_pred, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues",
        colorbar=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_plot(y_true, y_score, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="Test set")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def pr_plot(y_true, y_score, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="Test set")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def feature_importance_plot(pipeline, path: Path, title: str, top: int = 15) -> Path | None:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "coef_"):
        values = model.coef_.ravel()
        xlabel = "Coefficient (log-odds, scaled features)"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        xlabel = "Impurity-based importance"
    else:
        return None
    order = np.argsort(np.abs(values))[-top:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], values[order],
            color=["#E45756" if v > 0 else "#4C78A8" for v in values[order]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def model_comparison_plot(summary: dict, path: Path) -> Path:
    """Grouped bars of CV metrics (mean +/- std) for each model family."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = list(summary)
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(models):
        means = [summary[m]["cv"][f"{k}_mean"] for k in metrics]
        stds = [summary[m]["cv"][f"{k}_std"] for k in metrics]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3,
               label=m, color=colors[i % len(colors)])
    ax.set_xticks(x, metrics)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-fold CV score")
    ax.set_title("Model comparison (cross-validation, mean ± std)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
