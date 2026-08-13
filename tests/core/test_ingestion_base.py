"""Interface-contract tests for `core.ingestion.base.Ingestor`.

These tests confirm the *contract* is enforceable (can't instantiate
the ABC directly, a minimal concrete subclass must implement both
abstract methods, and the concrete `ingest()` convenience wires
`fetch()`/`load()` together) — not a full implementation test of any
real ingestor.
"""

from __future__ import annotations

import pytest

from core.ingestion.base import Ingestor


def test_cannot_instantiate_abstract_ingestor():
    with pytest.raises(TypeError):
        Ingestor()  # type: ignore[abstract]


def test_subclass_missing_fetch_cannot_instantiate():
    class MissingFetch(Ingestor):
        def load(self):
            return "data"

    with pytest.raises(TypeError):
        MissingFetch()  # type: ignore[abstract]


def test_subclass_missing_load_cannot_instantiate():
    class MissingLoad(Ingestor):
        def fetch(self):
            pass

    with pytest.raises(TypeError):
        MissingLoad()  # type: ignore[abstract]


def test_minimal_concrete_subclass_satisfies_contract():
    calls = []

    class FakeIngestor(Ingestor):
        def fetch(self):
            calls.append("fetch")

        def load(self):
            calls.append("load")
            return {"rows": 3}

    ingestor = FakeIngestor()
    result = ingestor.ingest()

    assert calls == ["fetch", "load"]
    assert result == {"rows": 3}


def test_load_called_independently_of_fetch():
    class FakeIngestor(Ingestor):
        def fetch(self):
            pass

        def load(self):
            return "loaded"

    ingestor = FakeIngestor()
    assert ingestor.load() == "loaded"
