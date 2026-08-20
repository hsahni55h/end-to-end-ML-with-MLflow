# notebooks/

EDA notebooks only — one subfolder per project, matching the naming used
under `projects/` (e.g. `01_wine_quality/`, `02_fraud_graph/` once it
exists).

## What belongs here

Exploration only: shape/dtypes checks, missing values, target
distribution, correlations, outlier inspection, and whatever else is
useful for understanding a dataset before building its pipeline. No
pipeline logic — that belongs in `core/` (shared, dataset-agnostic base
classes) or `projects/<name>/` (the actual ingestion/validation/
transformation/training/evaluation code). If a notebook cell is doing
something a pipeline stage should own, it doesn't belong here long-term.

## Workflow

For each new project, EDA in `notebooks/<project_name>/` happens *before*
that project's implementation session — not alongside it, and not after.
The point is for exploration findings (which features need scaling,
whether there are missing values or outliers to handle, what the target
distribution looks like, which features correlate with the target or
each other) to inform the pipeline's actual design decisions, rather than
building the pipeline first and only checking the data afterward.

## Starting exploration for a new dataset

1. Create `notebooks/<project_name>/` (matching the project's folder name
   under `projects/`).
2. Add a numbered EDA notebook, e.g. `01_eda.ipynb`.
3. Add a short `notebooks/<project_name>/README.md` describing what's in
   the folder and what it's exploring (see
   `notebooks/01_wine_quality/README.md` for the pattern).
4. Do the exploration, then summarize the key findings in that project's
   notebook README before starting the pipeline implementation session.
