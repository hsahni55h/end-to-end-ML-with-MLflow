"""Interface-contract tests for `core.evaluation.base.Evaluator`.

Confirms the ABC can't be instantiated directly, a minimal concrete
subclass must implement `evaluate()`, and the concrete `log_metrics()`
default persists a local JSON file and calls MLflow logging functions
(mocked here — these tests don't want a real MLflow run as a side
effect).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from core.evaluation.base import Evaluator


def test_cannot_instantiate_abstract_evaluator():
    with pytest.raises(TypeError):
        Evaluator()  # type: ignore[abstract]


def test_subclass_missing_evaluate_cannot_instantiate():
    class MissingEvaluate(Evaluator):
        pass

    with pytest.raises(TypeError):
        MissingEvaluate()  # type: ignore[abstract]


def test_minimal_concrete_subclass_satisfies_contract():
    class FakeEvaluator(Evaluator):
        def evaluate(self, model, test_data):
            return {"rmse": 0.5, "r2": 0.9}

    evaluator = FakeEvaluator()
    metrics = evaluator.evaluate(model=object(), test_data=object())

    assert metrics == {"rmse": 0.5, "r2": 0.9}


def test_log_metrics_persists_local_json(tmp_path):
    class FakeEvaluator(Evaluator):
        def evaluate(self, model, test_data):
            return {}

    metrics_file = tmp_path / "metrics.json"
    evaluator = FakeEvaluator()

    with patch("core.evaluation.base.mlflow") as mock_mlflow:
        evaluator.log_metrics({"rmse": 0.5}, metrics_file=metrics_file)

    assert metrics_file.exists()
    assert json.loads(metrics_file.read_text()) == {"rmse": 0.5}
    mock_mlflow.log_metric.assert_called_once_with("rmse", 0.5)


def test_log_metrics_calls_mlflow_log_params_and_metrics():
    class FakeEvaluator(Evaluator):
        def evaluate(self, model, test_data):
            return {}

    evaluator = FakeEvaluator()

    with patch("core.evaluation.base.mlflow") as mock_mlflow:
        evaluator.log_metrics(
            {"rmse": 0.5, "mae": 0.3}, params={"alpha": 0.4, "l1_ratio": 0.3}
        )

    mock_mlflow.log_params.assert_called_once_with({"alpha": 0.4, "l1_ratio": 0.3})
    assert mock_mlflow.log_metric.call_count == 2


def test_log_metrics_with_no_artifacts_is_a_noop_for_artifact_logging():
    class FakeEvaluator(Evaluator):
        def evaluate(self, model, test_data):
            return {}

    evaluator = FakeEvaluator()

    with patch("core.evaluation.base.mlflow") as mock_mlflow:
        evaluator.log_metrics({"rmse": 0.5})

    mock_mlflow.log_artifact.assert_not_called()
    mock_mlflow.log_dict.assert_not_called()


def test_log_metrics_logs_path_artifact_via_mlflow_log_artifact(tmp_path):
    class FakeEvaluator(Evaluator):
        def evaluate(self, model, test_data):
            return {}

    sample_image = tmp_path / "sample_prediction.png"
    sample_image.write_bytes(b"fake-image-bytes")
    evaluator = FakeEvaluator()

    with patch("core.evaluation.base.mlflow") as mock_mlflow:
        evaluator.log_metrics({}, artifacts={"sample_prediction": sample_image})

    mock_mlflow.log_artifact.assert_called_once_with(
        str(sample_image), artifact_path="sample_prediction"
    )


def test_log_metrics_logs_dict_artifact_via_mlflow_log_dict():
    class FakeEvaluator(Evaluator):
        def evaluate(self, model, test_data):
            return {}

    trace = {"question": "what is x?", "answer": "x is y", "faithful": True}
    evaluator = FakeEvaluator()

    with patch("core.evaluation.base.mlflow") as mock_mlflow:
        evaluator.log_metrics({}, artifacts={"llm_judge_trace": trace})

    mock_mlflow.log_dict.assert_called_once_with(trace, "llm_judge_trace.json")


def test_log_metrics_rejects_unsupported_artifact_type():
    class FakeEvaluator(Evaluator):
        def evaluate(self, model, test_data):
            return {}

    evaluator = FakeEvaluator()

    with patch("core.evaluation.base.mlflow"):
        with pytest.raises(TypeError):
            evaluator.log_metrics({}, artifacts={"weird": object()})
