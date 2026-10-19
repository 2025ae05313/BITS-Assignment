"""Central configuration: paths, column schema and constants.

Keeping the schema in one place means the data cleaning code, the training
pipeline, the API request model and the tests all agree on feature names.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("HEARTML_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart_cleveland_raw.csv"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "heart_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow: SQLite backend (supports the model registry) + local artifact folder.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "heart-disease-classification")
REGISTERED_MODEL_NAME = "heart-disease-classifier"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Official UCI source (processed Cleveland subset: 303 patients, 14 attributes).
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# Fallback mirror of the same 303 Cleveland records (ISLR "Heart" format).
MIRROR_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"

# UCI column order for processed.cleveland.data
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous / count features -> median impute + standard scaling
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Coded categorical features -> mode impute + one-hot encoding
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plausible clinical ranges; values outside are treated as missing (data errors).
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
    "ca": (0, 3),
}
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}

# Human-readable descriptions used in the report and API docs
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thallium stress test (3 normal, 6 fixed defect, 7 reversible defect)",
}
