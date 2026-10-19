"""Unit tests for the preprocessing transformer."""
import numpy as np
import pandas as pd

from heartml import config
from heartml.features import build_preprocessor


def _expected_width(df):
    n_cat = sum(df[c].nunique() for c in config.CATEGORICAL_FEATURES)
    return len(config.NUMERIC_FEATURES) + n_cat


def test_output_has_no_missing_values(clean_df):
    X = clean_df[config.FEATURES]
    assert X.isna().any().any(), "fixture should contain NaNs"
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()


def test_output_width_matches_one_hot_encoding(clean_df):
    X = clean_df[config.FEATURES]
    pre = build_preprocessor().fit(X)
    assert pre.transform(X).shape == (len(X), _expected_width(X.dropna()))
    names = pre.get_feature_names_out()
    assert "age" in names and "cp_4.0" in names


def test_numeric_features_are_standardised(clean_df):
    X = clean_df[config.FEATURES]
    Xt = build_preprocessor().fit_transform(X)
    num = Xt[:, : len(config.NUMERIC_FEATURES)]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num.std(axis=0), 1, atol=1e-6)


def test_unseen_category_does_not_crash(clean_df):
    X = clean_df[config.FEATURES]
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["restecg"] = 9  # never seen during fit
    out = pre.transform(row)
    assert out.shape[1] == pre.transform(X.iloc[[0]]).shape[1]


def test_transform_is_deterministic(clean_df):
    X = clean_df[config.FEATURES]
    pre = build_preprocessor().fit(X)
    pd.testing.assert_frame_equal(pd.DataFrame(pre.transform(X)), pd.DataFrame(pre.transform(X)))
