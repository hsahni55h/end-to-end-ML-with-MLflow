"""wine_quality Transformer: outlier clipping + scaling feature engineering.

Subclasses `core.transformation.base.Transformer[pd.DataFrame]`. The legacy
src/mlProject pipeline only did a train/test split with no real feature
engineering — this is the real implementation the base class's fit/transform
split was designed for.

No categorical encoding step: every column in schema.yaml's COLUMNS is
already numeric for this dataset (UCI winequality-red), so there is nothing
to encode. Called out explicitly rather than silently omitted, per this
session's scope ("encoding if applicable").
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.transformation.base import Transformer


class WineQualityTransformer(Transformer[pd.DataFrame]):
    """Clips per-feature outliers (IQR rule) then standard-scales features.

    `fit()` learns clip bounds and scaler statistics from the data it's
    given — call it with train data only, then `transform()` both train
    and test with the same fitted bounds/scaler, to avoid leaking test-set
    statistics into training (the reason `Transformer` separates fit/transform
    in the first place).
    """

    def __init__(self, target_column: str, iqr_multiplier: float = 1.5) -> None:
        self.target_column = target_column
        self.iqr_multiplier = iqr_multiplier
        self.feature_columns: Optional[List[str]] = None
        self._bounds: Optional[pd.DataFrame] = None
        self._scaler: Optional[StandardScaler] = None

    def fit(self, data: pd.DataFrame) -> "WineQualityTransformer":
        self.feature_columns = [c for c in data.columns if c != self.target_column]
        features = data[self.feature_columns]

        q1 = features.quantile(0.25)
        q3 = features.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        self._bounds = pd.DataFrame({"lower": lower, "upper": upper})

        clipped = features.clip(lower=lower, upper=upper, axis=1)
        self._scaler = StandardScaler()
        self._scaler.fit(clipped)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or self._bounds is None or self.feature_columns is None:
            raise RuntimeError(
                "WineQualityTransformer must be fit() before transform()."
            )

        features = data[self.feature_columns]
        clipped = features.clip(
            lower=self._bounds["lower"], upper=self._bounds["upper"], axis=1
        )
        scaled = pd.DataFrame(
            self._scaler.transform(clipped),
            columns=self.feature_columns,
            index=data.index,
        )
        scaled[self.target_column] = data[self.target_column].values
        return scaled
