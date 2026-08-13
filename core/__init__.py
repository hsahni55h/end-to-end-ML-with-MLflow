"""Shared, dataset-agnostic pipeline engine.

Subfolders (`ingestion/`, `validation/`, `transformation/`, `training/`,
`evaluation/`) each hold an abstract base class in `base.py` that every
`projects/*/` pipeline stage subclasses. See each `base.py` module
docstring for the interface contract and how it's expected to
generalize across current and future projects.
"""
