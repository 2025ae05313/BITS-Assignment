"""Request/response contracts for the prediction API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """One patient record. Ranges mirror the cleaning rules used in training.

    `ca` and `thal` are optional: the model pipeline imputes them, exactly as it
    did for the few missing values in the UCI data.
    """

    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=70, le=220, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=60, le=220, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression (exercise vs rest)")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed, 7 reversible")

    @classmethod
    def example(cls) -> dict:
        return {
            "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6,
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease risk, 0 = no disease")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float = Field(..., description="P(heart disease)")
    risk_band: Literal["low", "moderate", "high"]
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    instances: list[PatientFeatures] = Field(..., min_length=1, max_length=256)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
