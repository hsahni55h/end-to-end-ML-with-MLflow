"""Interface-contract tests for `core.transformation.base.Transformer`.

Confirms the ABC can't be instantiated directly, a minimal concrete
subclass must implement `fit()`/`transform()`, `fit_transform()` wires
them together, and the concrete `split()` default behaves as a
tabular-appropriate random split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.transformation.base import Transformer


def test_cannot_instantiate_abstract_transformer():
    with pytest.raises(TypeError):
        Transformer()  # type: ignore[abstract]


def test_subclass_missing_transform_cannot_instantiate():
    class MissingTransform(Transformer):
        def fit(self, data):
            return self

    with pytest.raises(TypeError):
        MissingTransform()  # type: ignore[abstract]


def test_fit_transform_wires_fit_and_transform():
    calls = []

    class FakeTransformer(Transformer):
        def fit(self, data):
            calls.append("fit")
            return self

        def transform(self, data):
            calls.append("transform")
            return data * 2

    transformer = FakeTransformer()
    result = transformer.fit_transform(5)

    assert calls == ["fit", "transform"]
    assert result == 10


def test_fit_returns_self_for_chaining():
    class FakeTransformer(Transformer):
        def fit(self, data):
            return self

        def transform(self, data):
            return data

    transformer = FakeTransformer()
    assert transformer.fit(None) is transformer


def test_default_split_returns_two_partitions():
    class FakeTransformer(Transformer):
        def fit(self, data):
            return self

        def transform(self, data):
            return data

    df = pd.DataFrame({"a": range(20)})
    transformer = FakeTransformer()

    train, test = transformer.split(df, test_size=0.25, random_state=42)

    assert len(train) == 15
    assert len(test) == 5
