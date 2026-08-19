"""wine_quality prediction pipeline for the FastAPI serving layer.

Inlined from the retired `app.py` (Flask) reference implementation per
S1.3 scope: same `joblib.load` + `.predict()` logic, no other change.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class PredictionPipeline:
    """Loads a trained wine_quality model and serves predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model = joblib.load(model_path)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)
