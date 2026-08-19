"""Pydantic request/response schemas for the wine_quality serving API.

Field names/order match `projects/01_wine_quality/config/schema.yaml`'s
`COLUMNS` (minus the target `quality`), which is also the exact column
order `WineQualityTrainer`/`WineQualityEvaluator` train and evaluate
against. Field names use underscores instead of the raw dataset's spaces
(e.g. `fixed_acidity` for "fixed acidity") since Pydantic/Python
attribute names can't contain spaces — this mirrors the form-field naming
`app.py` already used.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


class WineQualityFeatures(BaseModel):
    """One row of wine_quality model input features."""

    fixed_acidity: float = Field(..., description="Fixed acidity (g/dm^3)")
    volatile_acidity: float = Field(..., description="Volatile acidity (g/dm^3)")
    citric_acid: float = Field(..., description="Citric acid (g/dm^3)")
    residual_sugar: float = Field(..., description="Residual sugar (g/dm^3)")
    chlorides: float = Field(..., description="Chlorides (g/dm^3)")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide (mg/dm^3)")
    total_sulfur_dioxide: float = Field(
        ..., description="Total sulfur dioxide (mg/dm^3)"
    )
    density: float = Field(..., description="Density (g/cm^3)")
    pH: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates (g/dm^3)")
    alcohol: float = Field(..., description="Alcohol (% by volume)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        }
    )


class WineQualityPrediction(BaseModel):
    """Response returned by `/predict`."""

    predicted_quality: float = Field(
        ..., description="Predicted wine quality score from the trained model"
    )


class TrainingResponse(BaseModel):
    """Response returned by `/train` — the test-set metrics from the run."""

    metrics: Dict[str, float] = Field(
        ..., description="Test-set metrics (rmse, mae, r2) from this training run"
    )
