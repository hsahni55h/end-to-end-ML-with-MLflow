# production-ml-platform

## What this is

A single, reusable platform for taking machine learning problems from raw
data to a served prediction — the same structure whether the underlying
problem is tabular regression, graph-based fraud detection, relational
credit risk, time series forecasting, computer vision, or an LLM/RAG
pipeline. Instead of writing a one-off script per dataset, every project
plugs into the same ingestion → validation → feature engineering → tuning
→ tracking → serving pipeline shape, implemented once and reused.

The aim: prove that a shared core can genuinely generalize across very
different kinds of ML problems, not just tabular CSVs — and do it with
the practices a real ML platform needs (config-driven pipelines,
experiment tracking, data versioning, typed serving, tests), not just
notebook-quality code.

## How it's built

A shared `core/` engine defines the pipeline once, as a set of abstract
base classes; each dataset/problem gets its own folder under `projects/`
that subclasses those base classes and overrides only what's genuinely
domain-specific (feature engineering, model choice, metrics). New
projects are added by writing a `projects/<name>/` that plugs into the
existing `core/` — not by writing a new pipeline from scratch, and not by
working around a base class that doesn't fit (if that happens, the base
class itself gets extended).

## Architecture

- **`core/`** — abstract, dataset-agnostic pipeline stages:
  - `core/ingestion/` — `Ingestor`: fetch/load raw data from a source
    (file, URL, database, API).
  - `core/validation/` — `Validator`: check ingested data against an
    expected schema, record pass/fail status.
  - `core/transformation/` — `Transformer`: fit/transform-style feature
    engineering (scaling, encoding, outlier handling), with fit and
    transform kept separate to avoid leaking test-set statistics into
    training.
  - `core/training/` — `Trainer`: train (with hyperparameter tuning) and
    persist/load a model.
  - `core/evaluation/` — `Evaluator`: compute metrics and log them
    (params, scalar metrics, and arbitrary artifacts) to the experiment
    tracker.

  No project is allowed to bypass these to make its own code simpler — if
  an abstraction genuinely doesn't fit a new project, that's a `core/`
  design change, not a project-level workaround.

- **`projects/<name>/`** — one folder per ML problem, each subclassing
  every relevant `core/` base class and adding only what's
  domain-specific: its own `config.yaml`/`params.yaml`/`schema.yaml`,
  feature engineering, model choice, and metrics.

- **`serving/`** — a single FastAPI app (`serving/api/`), with one router
  and one set of typed Pydantic schemas (`serving/schemas/`) per project,
  serving predictions and exposing training as an endpoint.

- **`tests/`** — `tests/core/` (unit tests per base class),
  `tests/projects/` (integration tests per project pipeline),
  `tests/serving/` (API tests).

Planned, not yet built: `orchestration/` (scheduled/dependency-managed
pipeline runs) and `monitoring/` (drift and performance reports) as
top-level components, alongside more `projects/` covering graph, relational,
time series, computer vision, and LLM/RAG problems.

## The wine_quality project

`projects/01_wine_quality/` is the first and currently only project — a
regression problem (UCI Wine Quality) chosen deliberately as a simple,
well-understood dataset. Its purpose isn't the modeling problem itself;
it's the proof case that the `core/` framework — ingestion, validation,
real feature engineering, hyperparameter-tuned training, MLflow tracking,
and FastAPI serving — genuinely works end to end, on a dataset simple
enough that none of that infrastructure is obscured by domain complexity.
See [`projects/01_wine_quality/README.md`](projects/01_wine_quality/README.md)
for its approach and real results.

## Tech stack

| Concern | Tool |
|---|---|
| Language | Python 3.11 |
| Dependency management | [uv](https://github.com/astral-sh/uv) |
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow (+ DagsHub as a hosted remote) |
| Data/artifact versioning | DVC |
| Serving | FastAPI + Uvicorn, Pydantic schemas |
| Containerization | Docker |
| Testing | pytest |
| Linting/formatting | Ruff, Black |

Planned for later stages: Prefect (orchestration), Evidently AI
(monitoring), and — for the deep learning/GenAI projects — PyTorch,
HuggingFace, and a vector store (Chroma or Qdrant).

## Running it

See [`projects/01_wine_quality/README.md`](projects/01_wine_quality/README.md)
for that project's pipeline details, and the repo's [`Dockerfile`](Dockerfile)
for how the FastAPI serving layer is containerized (`docker build -t
production-ml-platform .` / `docker run -p 8080:8080 production-ml-platform`,
then open `/docs`). Dependencies are managed with **uv** — `uv sync` to
install, `uv run pytest` to run tests, `uv run uvicorn serving.api.main:app`
to run the API locally.

## MLflow / DagsHub

Experiment tracking uses MLflow. A DagsHub-hosted remote was used
previously but was deleted during a credential rotation early in the
project; a new one gets wired up when the next dataset project needs a
shared remote. Copy `.env.example` to `.env` and fill in credentials
if/when you configure your own remote — never hardcode them.



