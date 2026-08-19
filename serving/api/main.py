"""FastAPI serving app for the wine_quality project (S1.3).

Replaces the retired Flask `app.py`. Two endpoints:

- `POST /train`: invokes `projects.01_wine_quality.pipeline.run_pipeline()`
  directly (the real S1.2 training entrypoint) — no more `os.system(...)`
  shell-out to a deleted script.
- `POST /predict`: loads the model `run_pipeline()` just saved and returns
  a prediction, via the inlined `PredictionPipeline` (see `predictor.py`).

Both endpoints resolve the model path the same way, from
`projects/01_wine_quality/config/config.yaml`'s `model_trainer` section
via the pipeline module itself (`_load_yaml` + `PROJECT_ROOT`) — this is
the fix for the S1.2 follow-up gap where `/train` and `/predict` pointed
at two different `artifacts/` directories. The model is loaded fresh on
every `/predict` call (no caching at startup) so a `/train` triggered
without restarting the server is immediately visible to `/predict`,
matching `app.py`'s original per-request `PredictionPipeline()` behavior.

`01_wine_quality` starts with a digit, so it can't be named in a literal
`import`/`from ... import` statement — loaded via `importlib.import_module`
instead, same as `tests/projects/test_wine_quality_pipeline.py`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from serving.api.predictor import PredictionPipeline
from serving.schemas.wine_quality import (
    TrainingResponse,
    WineQualityFeatures,
    WineQualityPrediction,
)

app = FastAPI(
    title="production-ml-platform serving API",
    description="Serving layer for the wine_quality project (Phase 1, S1.3).",
    version="0.1.0",
)

_pipeline = importlib.import_module("projects.01_wine_quality.pipeline")


def _model_path() -> Path:
    """Resolve the model path from wine_quality's own config.yaml, so
    /predict always reads from wherever run_pipeline() actually writes."""
    config = _pipeline._load_yaml(_pipeline.PROJECT_ROOT / "config" / "config.yaml")
    model_cfg = config["model_trainer"]
    return _pipeline.PROJECT_ROOT / model_cfg["root_dir"] / model_cfg["model_name"]


@app.post("/predict", response_model=WineQualityPrediction)
def predict(features: WineQualityFeatures) -> WineQualityPrediction:
    model_path = _model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No trained model found at {model_path}. Call /train first.",
        )

    pipeline = PredictionPipeline(model_path)
    data = np.array(
        [
            [
                features.fixed_acidity,
                features.volatile_acidity,
                features.citric_acid,
                features.residual_sugar,
                features.chlorides,
                features.free_sulfur_dioxide,
                features.total_sulfur_dioxide,
                features.density,
                features.pH,
                features.sulphates,
                features.alcohol,
            ]
        ]
    )
    prediction = pipeline.predict(data)
    return WineQualityPrediction(predicted_quality=float(prediction[0]))


@app.post("/train", response_model=TrainingResponse)
def train() -> TrainingResponse:
    metrics = _pipeline.run_pipeline()
    return TrainingResponse(metrics=metrics)
