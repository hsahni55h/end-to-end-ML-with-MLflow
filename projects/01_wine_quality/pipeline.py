"""wine_quality pipeline: wires Ingestor -> Validator -> Transformer ->
Trainer -> Evaluator together end to end.

Run directly with:
    uv run python -m projects.01_wine_quality.pipeline

Note: this module's directory name (`01_wine_quality`) starts with a digit,
so it can't be referenced with a literal `import` statement from outside
the package (e.g. `from projects.01_wine_quality import pipeline` is a
syntax error) — use `importlib.import_module("projects.01_wine_quality.pipeline")`
instead, as `tests/projects/` does. Modules *inside* this package import
each other fine via relative imports (`from .ingestion import ...`), since
those don't need the package name to be a valid identifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlflow
import yaml

from .evaluator import WineQualityEvaluator
from .ingestion import WineQualityIngestor
from .trainer import WineQualityTrainer
from .transformation import WineQualityTransformer
from .validation import WineQualityValidator

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline() -> Dict[str, float]:
    """Run the full wine_quality pipeline once and return the test metrics."""
    config = _load_yaml(PROJECT_ROOT / "config.yaml")
    params = _load_yaml(PROJECT_ROOT / "params.yaml")
    schema = _load_yaml(PROJECT_ROOT / "schema.yaml")

    target_column = schema["TARGET_COLUMN"]["name"]

    # --- ingestion ---
    ing_cfg = config["data_ingestion"]
    ingestor = WineQualityIngestor(
        source_url=ing_cfg["source_URL"],
        local_data_file=PROJECT_ROOT / ing_cfg["local_data_file"],
        unzip_dir=PROJECT_ROOT / ing_cfg["unzip_dir"],
    )
    raw_data = ingestor.ingest()

    # --- validation ---
    val_cfg = config["data_validation"]
    validator = WineQualityValidator(
        status_file=PROJECT_ROOT / val_cfg["status_file"],
        schema=schema["COLUMNS"],
    )
    validation_result = validator.run(raw_data)
    if not validation_result.is_valid:
        raise ValueError(f"Data validation failed: {validation_result.errors}")

    # --- transformation (split first, fit on train only, to avoid leakage) ---
    split_cfg = params["train_test_split"]
    transformer = WineQualityTransformer(target_column=target_column)
    train_raw, test_raw = transformer.split(
        raw_data,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    train_data = transformer.fit_transform(train_raw)
    test_data = transformer.transform(test_raw)

    trans_cfg = config["data_transformation"]
    train_dir = PROJECT_ROOT / trans_cfg["root_dir"]
    train_dir.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_dir / "train.csv", index=False)
    test_data.to_csv(train_dir / "test.csv", index=False)

    # --- training + evaluation, wrapped in a single MLflow run ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    optuna_cfg = params["optuna"]
    trainer = WineQualityTrainer(
        target_column=target_column,
        n_trials=optuna_cfg["n_trials"],
        val_size=optuna_cfg["val_size"],
        random_state=split_cfg["random_state"],
    )
    evaluator = WineQualityEvaluator(target_column=target_column)

    model_cfg = config["model_trainer"]
    model_path = PROJECT_ROOT / model_cfg["root_dir"] / model_cfg["model_name"]

    eval_cfg = config["model_evaluation"]
    metrics_path = PROJECT_ROOT / eval_cfg["metric_file_name"]

    with mlflow.start_run(run_name="wine_quality_elasticnet"):
        model = trainer.train(train_data)
        trainer.save(model, model_path)

        metrics = evaluator.evaluate(model, test_data)
        evaluator.log_metrics(
            metrics,
            params=trainer.best_params_,
            metrics_file=metrics_path,
            artifacts={"optuna_best_params": trainer.best_params_},
        )
        # Not calling mlflow.sklearn.log_model() here: it logs a whole
        # artifact *directory* (MLmodel/, conda.yaml, python_env.yaml, ...)
        # via mlflow's LocalArtifactRepository.log_artifacts(), which uses
        # `distutils.dir_util.copy_tree()` internally in mlflow==2.2.2
        # (pinned, legacy). That code path is broken in this repo's actual
        # environment (setuptools<81's vendored distutils shim is missing
        # `distutils._modified`, needed by `dir_util.copy_tree`) — a real
        # mlflow-vs-modern-toolchain incompatibility, not a core/ design
        # issue. `Evaluator.log_metrics()`'s `artifacts` param logs single
        # files via `mlflow.log_artifact`/`log_dict` instead, which use
        # `shutil.copyfile` and are unaffected. The trained model is still
        # persisted via `Trainer.save()` (joblib) above. Revisit
        # `log_model()` when mlflow is upgraded (PLAN.md Phase 1 task 5).

    return metrics


if __name__ == "__main__":
    run_pipeline()
