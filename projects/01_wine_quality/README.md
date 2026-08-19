# 01 — Wine Quality

## Problem

Predict red wine quality (a 0-10 score, modeled as regression) from 11
physicochemical measurements (acidity, sugar, sulphates, alcohol, etc.).
The dataset itself is a well-known "tutorial" one — the point of this
project isn't the modeling problem, it's proving out the `core/` base
class framework (`Ingestor`/`Validator`/`Transformer`/`Trainer`/`Evaluator`)
end-to-end, including real feature engineering, hyperparameter tuning, MLflow
tracking, and FastAPI serving, on a dataset simple enough that none of that
infrastructure work is obscured by domain complexity. Later projects
(`projects/02_*` onward) reuse this same framework on harder, less
generic datasets.

## Dataset

[UCI Wine Quality (red)](https://archive.ics.uci.edu/dataset/186/wine+quality)
— 1,599 rows, 11 numeric physicochemical features, one integer target
(`quality`, 3-8 in this subset). Fetched from a zip mirrored in
[hsahni55h/end-to-end-ML-with-MLflow](https://github.com/hsahni55h/end-to-end-ML-with-MLflow)
(this repo's predecessor); the zip is committed locally under
`data/raw/winequality-data.zip` and DVC-tracked (see Limitations), so
ingestion runs network-free from a fresh clone.

## Approach

- **Ingestion / validation**: [`ingestion.py`](ingestion.py) unzips the
  local data file; [`validation.py`](validation.py) checks the ingested
  columns against `config/schema.yaml` and writes a pass/fail status file.
- **Feature engineering** ([`transformation.py`](transformation.py)): the
  predecessor pipeline did a plain train/test split with no real
  transformation step. This project adds IQR-based outlier clipping
  followed by `StandardScaler`, both fit on the train split only and
  applied to train/test with the same fitted bounds — avoiding test-set
  leakage, which is the reason `core.transformation.base.Transformer`
  separates `fit()` from `transform()` in the first place. No categorical
  encoding: every feature column is already numeric for this dataset, so
  there's nothing to encode (called out explicitly rather than silently
  skipped).
- **Model + tuning** ([`trainer.py`](trainer.py)): `ElasticNet`, tuned over
  `alpha` and `l1_ratio` with Optuna (20 trials per run, TPE sampler,
  minimizing validation RMSE on an internal split of the training data).
  Each trial runs inside a nested MLflow run via `mlflow.sklearn.autolog()`,
  so every trial's params/metrics are tracked individually; the final model
  is refit on the full training set with the best-found hyperparameters.
  This replaces the predecessor's hardcoded `alpha=0.4, l1_ratio=0.3`.
- **What didn't work / was dropped**: `mlflow.sklearn.log_model()` is
  unusable against this repo's current toolchain — it hits a
  `distutils`/`setuptools` incompatibility with `mlflow==2.2.2` (pinned,
  to be revisited on a future MLflow upgrade). The model is persisted
  instead via `Trainer.save()`/`joblib`, and MLflow only tracks
  metrics/params, not a logged model artifact — a known gap, not an
  oversight.
- **Serving**: [`serving/api/`](../../serving/api/) exposes `/train` (runs
  this pipeline) and `/predict` (loads the joblib model this pipeline
  wrote) over FastAPI. See the root [README.md](../../README.md) for how
  to run it.

## Results

Optuna's sampler isn't seeded, so each `/train` (or `run_pipeline()`) run
tunes to slightly different hyperparameters and metrics. Three verified,
real (non-test) runs so far, all on the same train/test split:

| Run | RMSE | MAE | R² | alpha | l1_ratio |
|---|---|---|---|---|---|
| S1.2 (initial `core/` migration) | 0.628 | 0.506 | 0.363 | ≈0.052 | ≈0.509 |
| S1.2 follow-up (post data-path move) | 0.638 | — | — | ≈0.621 | — |
| S1.3 (via `/train` endpoint) | 0.642 | 0.522 | 0.334 | — | — |

No baseline metric was recorded for the predecessor's hardcoded
`alpha=0.4, l1_ratio=0.3` model, so it isn't included as a baseline row
above — only the tuned runs' own numbers are reported.

## What's next / limitations

- **Non-deterministic tuning**: `optuna.create_study()` isn't given a
  seeded sampler, so RMSE/MAE/R² and the winning hyperparameters vary
  run to run (see the three results rows above) — this is a real gap if
  reproducibility across runs matters, not just a cosmetic one.
- **`Trainer.train()` has no slot for tuning metadata**: the base class's
  `train() -> ModelT` signature only returns the model, not the
  hyperparameters that won the search. Worked around with a
  `WineQualityTrainer.best_params_` instance attribute (sklearn-style
  trailing underscore) rather than changing the base class. This gap is
  also visible at the API boundary — `TrainingResponse.metrics` in
  `serving/schemas/wine_quality.py` only surfaces `evaluate()`'s test
  metrics, not `best_params_`. Not promoted into a `core/` change yet;
  flagged here for whenever a second project's tuning needs make it a
  recurring pattern worth generalizing.
- **No logged MLflow model artifact**: see "What didn't work" above —
  `mlflow.sklearn.log_model()` doesn't work against the current
  `mlflow==2.2.2`/`setuptools`/`scikit-learn` combination in this repo.
  Only params/metrics are tracked in MLflow; the actual model file lives
  in `artifacts/model_trainer/model.joblib`, versioned via DVC instead.
- **DVC remote not yet configured**: `dvc init` is done and
  `data/raw/winequality-data.zip` / `artifacts/` are DVC-tracked, but
  there's no remote — tracking is local-cache-only for now. A DagsHub
  remote is deferred to Phase 2 (the original DagsHub project was deleted
  during the Phase 0 credential rotation; a new one gets wired up when
  Phase 2's first dataset project needs it).
