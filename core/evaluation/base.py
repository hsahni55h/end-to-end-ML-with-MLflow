"""Abstract base class for model evaluation.

Evaluation scores a trained model against held-out data and logs the
result (locally as JSON, and to MLflow) for tracking/comparison across
runs.

wine_quality (current, tabular regression)
    evaluate(model, test_data) -> RMSE/MAE/R2
    log_metrics(metrics, params) -> mlflow.log_metric/log_param + a
        local metrics.json (see model_evaluation.py's `log_into_mlflow`)
    Subclass: RegressionEvaluator(Evaluator[ElasticNet])

Sketch — 05_defect_detection (CV, PyTorch, MVTec AD)
    evaluate(model, test_data) -> precision/recall/AUROC on the anomaly
        class; also wants to log sample prediction images via
        mlflow.log_image(), which plain scalar-metric logging doesn't
        cover
    Subclass: DefectDetectionEvaluator(Evaluator[torch.nn.Module])
        overrides log_metrics() to additionally log a handful of
        sample prediction images

Sketch — 06_rag_eval (LLM/RAG)
    evaluate(model, test_data) -> retrieval recall@k + answer
        faithfulness/groundedness against a hand-built QA eval set.
        `model` here is a retrieval+generation pipeline, not a fitted
        estimator, and scoring faithfulness may require an
        LLM-as-judge call (network I/O) rather than a pure local
        computation
    Subclass: RagEvaluator(Evaluator[RagPipeline])

Generalizes: `evaluate() -> dict[str, float]` plus a separate
`log_metrics()` step holds across all three — every domain ultimately
produces a named set of scalar metrics, even though *how* those metrics
are computed differs enormously (sklearn metric functions vs.
LLM-as-judge calls). Typing `model`/`test_data` as generic (`ModelT`/
`T`) rather than assuming an sklearn-style estimator + `(X, y)` pair is
what avoids tabular-regression lock-in here.

Flag (partially addressed): `log_metrics()` originally only covered
scalar metrics + params + a local JSON dump. It now also accepts an
optional `artifacts` mapping so 05_defect_detection can pass sample
prediction images and 06_rag_eval can pass LLM-judge trace data,
without forcing every project through a scalar-only signature.
`artifacts` defaults to `None` (a no-op), so wine_quality's usage is
unaffected. The default handling only covers file-path and dict-shaped
artifacts (see `log_metrics()` docstring) — image objects, tensors, and
other exotic types still need a subclass override. Revisit further once
05_defect_detection or 06_rag_eval is actually implemented.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic, Optional, TypeVar

import mlflow

T = TypeVar("T")
ModelT = TypeVar("ModelT")


class Evaluator(ABC, Generic[T, ModelT]):
    """Abstract contract for scoring a trained model and logging the result.

    Subclasses implement `evaluate()`. `log_metrics()` has a concrete
    default (MLflow + local JSON) that covers the scalar-metric case,
    plus a best-effort default for path/dict-shaped artifacts; see the
    module docstring's flagged generalization gap for artifact types
    that still need a subclass override.
    """

    @abstractmethod
    def evaluate(self, model: ModelT, test_data: T) -> Dict[str, float]:
        """Score `model` against `test_data` and return named metrics."""
        raise NotImplementedError

    def log_metrics(
        self,
        metrics: Dict[str, float],
        params: Optional[Dict[str, object]] = None,
        metrics_file: Optional[Path] = None,
        artifacts: Optional[Dict[str, object]] = None,
    ) -> None:
        """Log `metrics` (and optional `params`) to MLflow, and persist
        `metrics` locally as JSON if `metrics_file` is given.

        Args:
            metrics: named scalar metrics to log.
            params: optional run parameters to log alongside the metrics.
            metrics_file: optional local path to also persist `metrics`
                as JSON.
            artifacts: optional mapping of artifact name -> artifact
                data, for logging things that aren't scalar metrics
                (e.g. sample prediction images for 05_defect_detection,
                LLM-judge trace data for 06_rag_eval). Default handling:
                `str`/`Path` values are logged as files via
                `mlflow.log_artifact`; `dict` values are logged as JSON
                via `mlflow.log_dict`. Left as `None` (the default), this
                is a no-op — wine_quality's usage is unaffected. Other
                artifact types (e.g. a PIL image or numpy array) aren't
                handled by this default; override `log_metrics()` in the
                subclass to log those via `mlflow.log_image()` or
                similar.
        """
        if metrics_file is not None:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        if params:
            mlflow.log_params(params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        if artifacts:
            for name, artifact in artifacts.items():
                if isinstance(artifact, (str, Path)):
                    mlflow.log_artifact(str(artifact), artifact_path=name)
                elif isinstance(artifact, dict):
                    mlflow.log_dict(artifact, f"{name}.json")
                else:
                    raise TypeError(
                        f"Unsupported artifact type for '{name}': "
                        f"{type(artifact).__name__}. Override log_metrics() "
                        "in the subclass to handle this artifact type."
                    )
