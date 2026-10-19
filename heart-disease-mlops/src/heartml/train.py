"""Train, tune, compare and package heart-disease classifiers with MLflow tracking.

Run tree (MLflow):
  experiment "heart-disease-classification"
    ├── baseline_majority_class            (sanity floor)
    ├── logistic_regression   (parent)     best params, CV + test metrics, plots, model
    │     └── lr_trial_XX     (children)   one run per grid point with its CV scores
    └── random_forest         (parent)
          └── rf_trial_XX     (children)

The model with the highest mean CV ROC-AUC is selected (the hold-out test set is
used only for reporting, never for selection), saved to models/ and registered.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import time
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from heartml import config, plots  # noqa: E402
from heartml.data import load_clean, split_xy  # noqa: E402
from heartml.features import build_pipeline  # noqa: E402

log = logging.getLogger("heartml.train")

# MLflow 3 serialises sklearn models with skops; whitelist the numpy type it flags.
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.tree._tree.Tree"]

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def model_specs(fast: bool = False) -> dict:
    """Candidate model families and their hyper-parameter grids."""
    if fast:  # tiny grids for CI smoke runs / unit tests
        return {
            "logistic_regression": (
                LogisticRegression(solver="liblinear", max_iter=2000,
                                   random_state=config.RANDOM_STATE),
                {"model__C": [0.1, 1.0], "model__l1_ratio": [0.0]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
                {"model__n_estimators": [100], "model__max_depth": [4]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=5000,
                               random_state=config.RANDOM_STATE),
            {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__l1_ratio": [0.0, 1.0],  # 0 = L2 (ridge), 1 = L1 (lasso)
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
    }


def holdout_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_score),
    }


def _short(params: dict) -> dict:
    return {k.replace("model__", ""): ("None" if v is None else v) for k, v in params.items()}


def _cv(n_splits: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def run_baseline(X_train, y_train, X_test, y_test, folds: int) -> dict:
    pipe = build_pipeline(DummyClassifier(strategy="most_frequent"))
    with mlflow.start_run(run_name="baseline_majority_class"):
        mlflow.set_tag("model_family", "baseline")
        cv = cross_validate(pipe, X_train, y_train, cv=_cv(folds), scoring=SCORING)
        metrics = {f"cv_{m}_mean": float(np.mean(cv[f"test_{m}"])) for m in SCORING}
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics.update(holdout_metrics(y_test, pipe.predict(X_test), proba))
        mlflow.log_params({"strategy": "most_frequent"})
        mlflow.log_metrics(metrics)
    return metrics


def tune_family(name, estimator, grid, X_train, y_train, X_test, y_test, folds, out_dir):
    """Grid-search one model family, log children + parent run, return summary."""
    pipe = build_pipeline(estimator)
    search = GridSearchCV(pipe, grid, cv=_cv(folds), scoring=SCORING, refit="roc_auc",
                          n_jobs=-1, return_train_score=True)
    prefix = "lr" if name.startswith("logistic") else "rf"

    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"model_family": name, "stage": "tuning"})
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        res = search.cv_results_

        # One nested child run per hyper-parameter combination
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{prefix}_trial_{i:02d}", nested=True):
                mlflow.set_tag("model_family", name)
                mlflow.log_params(_short(params))
                child = {f"cv_{m}_mean": float(res[f"mean_test_{m}"][i]) for m in SCORING}
                child.update({f"cv_{m}_std": float(res[f"std_test_{m}"][i]) for m in SCORING})
                child["cv_train_roc_auc_mean"] = float(res["mean_train_roc_auc"][i])
                child["rank_roc_auc"] = int(res["rank_test_roc_auc"][i])
                mlflow.log_metrics(child)

        best = search.best_index_
        cv_summary = {}
        for m in SCORING:
            cv_summary[f"{m}_mean"] = float(res[f"mean_test_{m}"][best])
            cv_summary[f"{m}_std"] = float(res[f"std_test_{m}"][best])
        model = search.best_estimator_
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tmetrics = holdout_metrics(y_test, pred, proba)
        overfit_gap = float(res["mean_train_roc_auc"][best] - res["mean_test_roc_auc"][best])

        mlflow.log_params({"best_" + k: v for k, v in _short(search.best_params_).items()})
        mlflow.log_params({"n_candidates": len(res["params"]), "cv_folds": folds,
                           "selection_metric": "roc_auc"})
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_summary.items()})
        mlflow.log_metrics(tmetrics)
        mlflow.log_metrics({"train_minus_cv_roc_auc": overfit_gap,
                            "tuning_seconds": fit_seconds})

        fam_dir = out_dir / name
        fam_dir.mkdir(parents=True, exist_ok=True)
        title = name.replace("_", " ").title()
        figs = [
            plots.confusion_matrix_plot(y_test, pred, fam_dir / "confusion_matrix.png",
                                        f"{title}: confusion matrix"),
            plots.roc_plot(y_test, proba, fam_dir / "roc_curve.png", f"{title}: ROC"),
            plots.pr_plot(y_test, proba, fam_dir / "pr_curve.png", f"{title}: precision-recall"),
            plots.feature_importance_plot(model, fam_dir / "feature_importance.png",
                                          f"{title}: top features"),
        ]
        for f in figs:
            if f is not None:
                mlflow.log_artifact(str(f), artifact_path="plots")

        cv_table = pd.DataFrame(res["params"]).rename(columns=lambda c: c.replace("model__", ""))
        for m in SCORING:
            cv_table[f"cv_{m}"] = res[f"mean_test_{m}"].round(4)
        cv_table = cv_table.sort_values("cv_roc_auc", ascending=False)
        cv_path = fam_dir / "cv_results.csv"
        cv_table.to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path), artifact_path="tables")

        signature = infer_signature(X_test, model.predict_proba(X_test))
        mlflow.sklearn.log_model(model, name="model", signature=signature,
                                 input_example=X_test.head(3),
                                 skops_trusted_types=SKOPS_TRUSTED)

        log.info("%s best %s | CV ROC-AUC %.3f ± %.3f | test ROC-AUC %.3f",
                 name, _short(search.best_params_), cv_summary["roc_auc_mean"],
                 cv_summary["roc_auc_std"], tmetrics["test_roc_auc"])
        return {
            "run_id": parent.info.run_id,
            "best_params": _short(search.best_params_),
            "cv": cv_summary,
            "test": tmetrics,
            "train_minus_cv_roc_auc": overfit_gap,
            "n_candidates": len(res["params"]),
            "model": model,
            "figures": [str(f) for f in figs if f is not None],
        }


def train(fast: bool = False, register: bool = True, folds: int = config.CV_FOLDS) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_clean(config.CLEAN_DATA_PATH)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    log.info("Train %d rows / test %d rows", len(X_train), len(X_test))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        config.MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT,
                                 artifact_location=config.MLFLOW_ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    out_dir = config.FIGURES_DIR / "models"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_baseline(X_train, y_train, X_test, y_test, folds)
    results = {}
    for name, (est, grid) in model_specs(fast).items():
        results[name] = tune_family(name, est, grid, X_train, y_train, X_test, y_test,
                                    folds, out_dir)

    # Selection on cross-validated ROC-AUC only (test set stays untouched)
    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc_mean"])
    best = results[best_name]
    log.info("Selected model: %s", best_name)

    comparison_fig = plots.model_comparison_plot(
        {k: v for k, v in results.items()}, out_dir / "model_comparison.png")

    # ---- Package the selected model -------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], config.MODEL_PATH)
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    metadata = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "model_version": version,
        "model_family": best_name,
        "best_params": best["best_params"],
        "cv_metrics": best["cv"],
        "test_metrics": best["test"],
        "decision_threshold": 0.5,
        "features": config.FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "training_rows": int(len(X_train)),
        "mlflow_run_id": best["run_id"],
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    # MLflow-format copy (portable: `mlflow models serve -m models/mlflow_model`)
    if config.MLFLOW_MODEL_DIR.exists():
        shutil.rmtree(config.MLFLOW_MODEL_DIR)
    signature = infer_signature(X_test, best["model"].predict_proba(X_test))
    mlflow.sklearn.save_model(best["model"], str(config.MLFLOW_MODEL_DIR),
                              signature=signature,
                              input_example=X_test.head(3),
                              skops_trusted_types=SKOPS_TRUSTED)

    # Summary run: comparison chart + selection decision
    with mlflow.start_run(run_name="model_selection"):
        mlflow.set_tags({"stage": "selection", "selected_family": best_name})
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("selected_run_id", best["run_id"])
        for name, r in results.items():
            mlflow.log_metric(f"{name}_cv_roc_auc", r["cv"]["roc_auc_mean"])
            mlflow.log_metric(f"{name}_test_roc_auc", r["test"]["test_roc_auc"])
        mlflow.log_metric("baseline_cv_accuracy", baseline["cv_accuracy_mean"])
        mlflow.log_artifact(str(comparison_fig), artifact_path="plots")
        mlflow.log_artifact(str(config.MODEL_METADATA_PATH), artifact_path="model_card")

    if register:
        try:
            mv = mlflow.register_model(f"runs:/{best['run_id']}/model",
                                       config.REGISTERED_MODEL_NAME)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion",
                                              mv.version)
            metadata["registry_version"] = int(mv.version)
            config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
            log.info("Registered %s v%s (alias: champion)", config.REGISTERED_MODEL_NAME,
                     mv.version)
        except Exception as err:  # registry unavailable (e.g. file store) - not fatal
            log.warning("Model registration skipped: %s", err)

    summary = {
        "selected_model": best_name,
        "baseline": baseline,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                   for k, v in results.items()},
        "metadata": metadata,
    }
    (config.REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train heart-disease classifiers")
    parser.add_argument("--fast", action="store_true", help="tiny grids (CI smoke run)")
    parser.add_argument("--no-register", action="store_true", help="skip model registry")
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    args = parser.parse_args(argv)
    summary = train(fast=args.fast, register=not args.no_register, folds=args.folds)
    sel = summary["selected_model"]
    m = summary["models"][sel]
    print(json.dumps({"selected_model": sel, "cv": m["cv"], "test": m["test"]}, indent=2))


if __name__ == "__main__":
    main()
