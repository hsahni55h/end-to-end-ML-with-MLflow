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

Flag: `log_metrics()` as written only covers scalar metrics + params +
a local JSON dump. 05_defect_detection needs to log sample images and
06_rag_eval may want to log full LLM-judge traces (prompt/response
pairs), not just numbers. Kept `log_metrics()` concrete (not abstract)
so those projects can extend/override it, but the *default*
implementation and signature only really fit the scalar-metric case.
Revisit the signature (e.g. an optional `artifacts: dict` param) once
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
    default (MLflow + local JSON) that covers the scalar-metric case;
    see the module docstring's flagged generalization gap for domains
    that need to log more than numbers.
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
    ) -> None:
        """Log `metrics` (and optional `params`) to MLflow, and persist
        `metrics` locally as JSON if `metrics_file` is given.

        Override to additionally log artifacts (images, LLM-judge
        traces, etc.) — see the module docstring's flagged
        generalization gap.
        """
        if metrics_file is not None:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        if params:
            mlflow.log_params(params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
