"""Shared fixtures. Tests never touch the network or the real data folder:
they build a synthetic UCI-shaped table so CI can run them before training."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartml import config
from heartml.features import build_pipeline


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """200 synthetic patients whose risk depends on a few real risk factors."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    logit = (0.04 * (df.age - 54) + 0.9 * (df.cp == 4) + 1.0 * df.exang
             + 0.6 * df.ca + 0.5 * df.oldpeak - 0.03 * (df.thalach - 150) - 1.5)
    df["num"] = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)),
                         rng.integers(1, 5, n), 0)
    df.loc[[3, 17], "ca"] = np.nan   # mimic the UCI '?' entries
    df.loc[[5], "thal"] = np.nan
    return df


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from heartml.data import clean
    return clean(raw_df)


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, clean_df):
    """A small fitted pipeline + metadata, saved the same way train.py does."""
    out = tmp_path_factory.mktemp("model")
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    joblib.dump(pipe, out / "model.joblib")
    (out / "model_metadata.json").write_text(json.dumps({
        "model_version": "test-0001", "model_family": "logistic_regression",
        "decision_threshold": 0.5, "features": config.FEATURES,
        "best_params": {}, "cv_metrics": {}, "test_metrics": {},
    }))
    return out
