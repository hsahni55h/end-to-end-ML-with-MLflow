"""wine_quality Evaluator: RMSE/MAE/R2 on held-out test data.

Subclasses `core.evaluation.base.Evaluator[pd.DataFrame, ElasticNet]`. Only
`evaluate()` is overridden — `log_metrics()` is used as-is from the base
class (MLflow logging + local JSON + the `artifacts` param added in S1.1),
which is exactly what this session needs to confirm end-to-end.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.evaluation.base import Evaluator


class WineQualityEvaluator(Evaluator[pd.DataFrame, ElasticNet]):
    """Scores a fitted ElasticNet model against held-out test data."""

    def __init__(self, target_column: str) -> None:
        self.target_column = target_column

    def evaluate(self, model: ElasticNet, test_data: pd.DataFrame) -> Dict[str, float]:
        X = test_data.drop(columns=[self.target_column])
        y = test_data[self.target_column]
        predictions = model.predict(X)

        rmse = float(np.sqrt(mean_squared_error(y, predictions)))
        mae = float(mean_absolute_error(y, predictions))
        r2 = float(r2_score(y, predictions))
        return {"rmse": rmse, "mae": mae, "r2": r2}
