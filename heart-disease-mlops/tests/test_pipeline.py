"""Integration tests: EDA and the full training run (fast grids) on synthetic data."""
import json

import joblib
import mlflow
import pytest

from heartml import config, train
from heartml.eda import run_eda


@pytest.fixture
def sandbox(tmp_path, monkeypatch, clean_df):
    """Redirect every output path to a temp dir so the real repo is untouched."""
    data = tmp_path / "heart_clean.csv"
    clean_df.to_csv(data, index=False)
    paths = {
        "CLEAN_DATA_PATH": data,
        "MODELS_DIR": tmp_path / "models",
        "MODEL_PATH": tmp_path / "models" / "model.joblib",
        "MODEL_METADATA_PATH": tmp_path / "models" / "model_metadata.json",
        "MLFLOW_MODEL_DIR": tmp_path / "models" / "mlflow_model",
        "REPORTS_DIR": tmp_path / "reports",
        "FIGURES_DIR": tmp_path / "reports" / "figures",
        "MLFLOW_ARTIFACT_ROOT": tmp_path / "mlruns",
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "MLFLOW_EXPERIMENT": "pytest-experiment",
    }
    for k, v in paths.items():
        monkeypatch.setattr(config, k, v)
    yield tmp_path
    mlflow.set_tracking_uri(None)


def test_eda_writes_all_figures(tmp_path, raw_df, clean_df):
    summary = run_eda(raw_df, clean_df, out_dir=tmp_path)
    assert summary["clean_rows"] == len(clean_df)
    assert len(list(tmp_path.glob("*.png"))) == 7
    assert (tmp_path / "eda_summary.json").exists()


def test_training_run_end_to_end(sandbox):
    summary = train.train(fast=True, register=True, folds=3)

    assert summary["selected_model"] in ("logistic_regression", "random_forest")
    model = joblib.load(config.MODEL_PATH)
    assert hasattr(model, "predict_proba")
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    assert meta["features"] == config.FEATURES
    assert meta["registry_version"] == 1
    assert (config.MLFLOW_MODEL_DIR / "MLmodel").exists()
    assert (config.FIGURES_DIR / "models" / "model_comparison.png").exists()

    runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT])
    # baseline + 2 parents + children (2 LR + 1 RF) + selection
    assert len(runs) == 1 + 2 + 3 + 1
    sel = runs[runs["tags.mlflow.runName"] == "model_selection"].iloc[0]
    assert sel["params.selected_model"] == summary["selected_model"]
