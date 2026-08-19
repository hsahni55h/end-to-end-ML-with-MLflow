"""wine_quality Trainer: Optuna-tuned ElasticNet, autologged to MLflow.

Subclasses `core.training.base.Trainer[pd.DataFrame, ElasticNet]`.

`train()` expects to be called inside an active MLflow run (see
`pipeline.run_pipeline()`) — each Optuna trial opens a *nested* run under
that active run via `mlflow.start_run(nested=True)`, so `mlflow.sklearn.
autolog()` captures every trial's params/metrics as its own child run,
without requiring manual `mlflow.log_param`/`log_metric` calls per trial.

Tuning splits `train_data` further into an internal train/validation split
(kept separate from the outer train/test split done by `Transformer.split`)
so trials are scored on unseen-during-that-trial data. Once the search
finishes, the final model is refit on the *full* `train_data` with the
best-found hyperparameters — that refit is autologged into the caller's
active (parent) run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from core.training.base import Trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


class WineQualityTrainer(Trainer[pd.DataFrame, ElasticNet]):
    """Tunes ElasticNet(alpha, l1_ratio) with Optuna, autologged to MLflow."""

    def __init__(
        self,
        target_column: str,
        n_trials: int = 20,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.target_column = target_column
        self.n_trials = n_trials
        self.val_size = val_size
        self.random_state = random_state
        self.best_params_: Optional[Dict[str, float]] = None
        self.study_: Optional[optuna.Study] = None

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        with mlflow.start_run(nested=True):
            model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state
            )
            model.fit(X_train, y_train)
            rmse = float(mean_squared_error(y_val, model.predict(X_val)) ** 0.5)
            mlflow.log_metric("val_rmse", rmse)

        return rmse

    def train(self, train_data: pd.DataFrame) -> ElasticNet:
        X = train_data.drop(columns=[self.target_column])
        y = train_data[self.target_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )

        # log_models=False: with n_trials nested runs, autologging a model
        # artifact per trial is mostly noise — the final refit below is
        # logged explicitly by the caller (see pipeline.run_pipeline()).
        # log_post_training_metrics=False: mlflow==2.2.2 (pinned, legacy —
        # see pyproject.toml) patches `sklearn.metrics.SCORERS` for this
        # feature, which was removed in the scikit-learn version this repo
        # otherwise uses (1.9). Disabling it avoids an AttributeError; the
        # metrics we actually care about (val_rmse per trial, rmse/mae/r2 in
        # WineQualityEvaluator) are logged explicitly regardless. Revisit
        # when mlflow is upgraded (PLAN.md Phase 1 task 5).
        mlflow.sklearn.autolog(
            log_models=False, log_post_training_metrics=False, silent=True
        )

        self.study_ = optuna.create_study(direction="minimize")
        self.study_.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
        )
        self.best_params_ = self.study_.best_params

        model = ElasticNet(random_state=self.random_state, **self.best_params_)
        model.fit(X, y)
        return model

    def save(self, model: ElasticNet, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    def load(self, path: Path) -> ElasticNet:
        return joblib.load(path)
