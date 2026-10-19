"""Feature engineering: one sklearn ColumnTransformer shared by training and serving."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartml import config


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numeric columns; mode-impute + one-hot encode codes.

    handle_unknown='ignore' means an unseen category at serving time produces an
    all-zero block instead of crashing the API.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Full model = preprocessing + classifier, persisted as one object."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])
