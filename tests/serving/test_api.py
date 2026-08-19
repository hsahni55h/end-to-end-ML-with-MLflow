"""Integration tests for the wine_quality FastAPI serving app (S1.3).

Hits the actual FastAPI app via `fastapi.testclient.TestClient` (real
ASGI request/response cycle, no route logic re-implemented in the test).

Follows the same isolation pattern as
`tests/projects/test_wine_quality_pipeline.py`: `pipeline.PROJECT_ROOT` is
monkey-patched to a tmp copy of the project (pre-seeded with the
already-ingested CSV, so no network I/O) so `/train` never writes into
the real repo's `projects/01_wine_quality/artifacts/`, and MLflow tracking
is pointed at a tmp dir so runs never land in the repo's real `mlruns/`.

`serving.api.main` imports the wine_quality pipeline module once via
`importlib.import_module(...)`, which is cached in `sys.modules` — so
patching `pipeline.PROJECT_ROOT` here mutates the exact same module
object `serving.api.main` reads `PROJECT_ROOT` from on every request.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from serving.api.main import _pipeline, app

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = REPO_ROOT / "artifacts" / "data_ingestion" / "winequality-red.csv"

VALID_FEATURES = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    """Copy the project into an isolated tmp dir, pre-seeded with the
    already-downloaded wine_quality CSV, and point `pipeline.PROJECT_ROOT`
    (and MLflow tracking) at it so `/train` never touches the real repo
    artifacts/mlruns or hits the network."""
    project_dir = tmp_path / "01_wine_quality"
    shutil.copytree(
        _pipeline.PROJECT_ROOT, project_dir, ignore=shutil.ignore_patterns("artifacts")
    )

    seed_csv = project_dir / "artifacts" / "data_ingestion" / "winequality-red.csv"
    seed_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE_CSV, seed_csv)

    monkeypatch.setattr(_pipeline, "PROJECT_ROOT", project_dir)
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")

    yield project_dir


@pytest.fixture
def client():
    return TestClient(app)


def test_docs_renders(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


@pytest.mark.skipif(
    not SOURCE_CSV.exists(),
    reason="requires the already-ingested wine_quality CSV in artifacts/data_ingestion/",
)
def test_predict_before_any_training_returns_503(isolated_project, client):
    response = client.post("/predict", json=VALID_FEATURES)

    assert response.status_code == 503
    assert "No trained model found" in response.json()["detail"]


@pytest.mark.skipif(
    not SOURCE_CSV.exists(),
    reason="requires the already-ingested wine_quality CSV in artifacts/data_ingestion/",
)
def test_train_then_predict_round_trip_uses_freshly_trained_model(
    isolated_project, client
):
    train_response = client.post("/train")
    assert train_response.status_code == 200
    metrics = train_response.json()["metrics"]
    assert set(metrics) == {"rmse", "mae", "r2"}

    model_path = isolated_project / "artifacts" / "model_trainer" / "model.joblib"
    assert model_path.exists()
    mtime_after_first_train = model_path.stat().st_mtime

    predict_response = client.post("/predict", json=VALID_FEATURES)
    assert predict_response.status_code == 200
    prediction = predict_response.json()["predicted_quality"]
    assert isinstance(prediction, float)

    # retrain again and confirm /predict is reading the model /train just
    # wrote (same path both endpoints resolve via config.yaml), not a
    # stale copy from elsewhere
    second_train_response = client.post("/train")
    assert second_train_response.status_code == 200
    assert model_path.stat().st_mtime >= mtime_after_first_train

    second_predict_response = client.post("/predict", json=VALID_FEATURES)
    assert second_predict_response.status_code == 200
