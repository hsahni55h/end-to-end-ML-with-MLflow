"""Integration tests for the wine_quality project (S1.2).

Covers the domain-specific pieces built this session (transformation,
validation, tuned training, evaluation) plus one full end-to-end run of
`projects.01_wine_quality.pipeline.run_pipeline()` confirming the whole
Ingestor -> Validator -> Transformer -> Trainer -> Evaluator stack works
together against `core/`'s base classes, including MLflow logging through
`Evaluator.log_metrics()` (scalar metrics + the `artifacts` param).

Note on imports: `01_wine_quality` starts with a digit, so it can't be
named in a literal `import`/`from ... import` statement (that's a syntax
error) — every module from the project is loaded via
`importlib.import_module()` instead, which has no such restriction.

Network access is avoided in the full end-to-end test: the real
winequality-red.csv already present in the repo's top-level
`artifacts/data_ingestion/` is copied into an isolated tmp copy of the
project before running, so `WineQualityIngestor.fetch()` finds the CSV
already there and short-circuits instead of hitting the network.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

transformation = importlib.import_module("projects.01_wine_quality.transformation")
validation = importlib.import_module("projects.01_wine_quality.validation")
trainer_module = importlib.import_module("projects.01_wine_quality.trainer")
evaluator_module = importlib.import_module("projects.01_wine_quality.evaluator")
pipeline = importlib.import_module("projects.01_wine_quality.pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = REPO_ROOT / "artifacts" / "data_ingestion" / "winequality-red.csv"

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"


def _make_synthetic_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(loc=5, scale=2, size=n) for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = rng.integers(3, 9, size=n)
    frame = pd.DataFrame(data)
    frame.loc[0, "alcohol"] = 500.0  # deliberate outlier
    return frame


@pytest.fixture(autouse=True)
def isolate_mlflow_tracking(tmp_path):
    """Every test in this module gets its own local MLflow tracking dir,
    so runs created here never land in the repo's real mlruns/."""
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    yield


def test_transformer_fit_transform_scales_and_clips_outliers():
    data = _make_synthetic_frame()
    transformer = transformation.WineQualityTransformer(target_column=TARGET_COLUMN)

    transformed = transformer.fit_transform(data)

    # the deliberate outlier was clipped before scaling, so it shouldn't
    # dominate the distribution after scaling
    assert transformed["alcohol"].iloc[0] < 10
    # target column passes through untouched
    assert transformed[TARGET_COLUMN].iloc[0] == data[TARGET_COLUMN].iloc[0]
    # scaled features are roughly standardized (mean ~0)
    assert abs(transformed["fixed acidity"].mean()) < 1e-6 + 0.5


def test_transformer_transform_before_fit_raises():
    data = _make_synthetic_frame()
    transformer = transformation.WineQualityTransformer(target_column=TARGET_COLUMN)

    with pytest.raises(RuntimeError):
        transformer.transform(data)


def test_validator_accepts_matching_schema_and_persists_status(tmp_path):
    schema = {col: "float64" for col in FEATURE_COLUMNS}
    schema[TARGET_COLUMN] = "int64"
    data = _make_synthetic_frame()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int64")

    validator = validation.WineQualityValidator(
        status_file=tmp_path / "status.txt", schema=schema
    )
    result = validator.run(data)

    assert result.is_valid
    assert "True" in (tmp_path / "status.txt").read_text()


def test_validator_flags_unexpected_column(tmp_path):
    schema = {col: "float64" for col in FEATURE_COLUMNS}
    schema[TARGET_COLUMN] = "int64"
    data = _make_synthetic_frame().rename(columns={"alcohol": "unexpected_column"})

    validator = validation.WineQualityValidator(
        status_file=tmp_path / "status.txt", schema=schema
    )
    result = validator.validate(data)

    assert not result.is_valid
    assert any("unexpected_column" in e for e in result.errors)


def test_trainer_returns_fitted_model_and_best_params():
    data = _make_synthetic_frame(n=80)
    trainer = trainer_module.WineQualityTrainer(
        target_column=TARGET_COLUMN, n_trials=2, val_size=0.25
    )

    with mlflow.start_run():
        model = trainer.train(data)

    assert trainer.best_params_ is not None
    assert {"alpha", "l1_ratio"} <= set(trainer.best_params_)
    assert hasattr(model, "predict")


def test_evaluator_returns_expected_metric_keys():
    data = _make_synthetic_frame(n=60)
    transformer = transformation.WineQualityTransformer(target_column=TARGET_COLUMN)
    transformed = transformer.fit_transform(data)

    trainer = trainer_module.WineQualityTrainer(target_column=TARGET_COLUMN, n_trials=1)
    with mlflow.start_run():
        model = trainer.train(transformed)

    evaluator = evaluator_module.WineQualityEvaluator(target_column=TARGET_COLUMN)
    metrics = evaluator.evaluate(model, transformed)

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())


@pytest.fixture
def isolated_project(tmp_path):
    """Copy the project into an isolated tmp dir, pre-seeded with the
    already-downloaded wine_quality CSV, and point pipeline.PROJECT_ROOT
    at it so run_pipeline() never touches the real repo artifacts/ or
    hits the network."""
    project_dir = tmp_path / "01_wine_quality"
    shutil.copytree(
        pipeline.PROJECT_ROOT, project_dir, ignore=shutil.ignore_patterns("artifacts")
    )

    seed_csv = project_dir / "artifacts" / "data_ingestion" / "winequality-red.csv"
    seed_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE_CSV, seed_csv)

    original_root = pipeline.PROJECT_ROOT
    pipeline.PROJECT_ROOT = project_dir
    try:
        yield project_dir
    finally:
        pipeline.PROJECT_ROOT = original_root


@pytest.mark.skipif(
    not SOURCE_CSV.exists(),
    reason="requires the already-ingested wine_quality CSV in artifacts/data_ingestion/",
)
def test_pipeline_runs_end_to_end_with_real_mlflow_run(isolated_project):
    metrics = pipeline.run_pipeline()

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())

    model_path = isolated_project / "artifacts" / "model_trainer" / "model.joblib"
    assert model_path.exists()

    metrics_path = isolated_project / "artifacts" / "model_evaluation" / "metrics.json"
    assert metrics_path.exists()

    status_path = isolated_project / "artifacts" / "data_validation" / "status.txt"
    assert status_path.exists()
    assert "True" in status_path.read_text()

    # confirm a real MLflow run was recorded with the expected metrics/params
    experiment = mlflow.get_experiment_by_name("wine_quality")
    assert experiment is not None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'wine_quality_elasticnet'",
    )
    assert len(runs) == 1
    assert "metrics.rmse" in runs.columns
    assert "params.alpha" in runs.columns
