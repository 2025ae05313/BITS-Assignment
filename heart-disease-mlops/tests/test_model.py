"""Unit tests for model training code."""
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from heartml import config
from heartml.features import build_pipeline
from heartml.train import holdout_metrics, model_specs


@pytest.fixture
def xy(clean_df):
    return clean_df[config.FEATURES], clean_df[config.TARGET]


def test_model_specs_define_two_families_with_grids():
    for fast in (False, True):
        specs = model_specs(fast)
        assert {"logistic_regression", "random_forest"} <= set(specs)
        for est, grid in specs.values():
            assert hasattr(est, "predict_proba")
            assert all(k.startswith("model__") for k in grid)


@pytest.mark.parametrize("family", ["logistic_regression", "random_forest"])
def test_each_family_fits_and_outputs_probabilities(xy, family):
    X, y = xy
    est, _ = model_specs(fast=True)[family]
    pipe = build_pipeline(est).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_beats_majority_baseline(xy):
    X, y = xy
    auc = cross_val_score(build_pipeline(LogisticRegression(max_iter=1000)), X, y,
                          cv=3, scoring="roc_auc").mean()
    base = cross_val_score(build_pipeline(DummyClassifier()), X, y, cv=3,
                           scoring="roc_auc").mean()
    assert auc > base + 0.1


def test_pipeline_handles_missing_inputs(xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    row = X.iloc[[0]].copy()
    row[["ca", "thal"]] = np.nan
    assert pipe.predict_proba(row).shape == (1, 2)


def test_saved_model_round_trips(tmp_path, xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), pipe.predict_proba(X))


def test_metric_helper():
    y = np.array([0, 0, 1, 1])
    m = holdout_metrics(y, np.array([0, 1, 1, 1]), np.array([0.1, 0.6, 0.7, 0.9]))
    assert m["test_recall"] == 1.0
    assert m["test_precision"] == pytest.approx(2 / 3)
    assert m["test_roc_auc"] == roc_auc_score(y, [0.1, 0.6, 0.7, 0.9])
