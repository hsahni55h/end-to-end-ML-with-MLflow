"""Interface-contract tests for `core.training.base.Trainer`.

Confirms the ABC can't be instantiated directly and that a minimal
concrete subclass must implement all three abstract methods
(`train()`, `save()`, `load()`).
"""

from __future__ import annotations

import pytest

from core.training.base import Trainer


def test_cannot_instantiate_abstract_trainer():
    with pytest.raises(TypeError):
        Trainer()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "missing_method",
    ["train", "save", "load"],
)
def test_subclass_missing_any_method_cannot_instantiate(missing_method):
    methods = {
        "train": lambda self, train_data: "model",
        "save": lambda self, model, path: None,
        "load": lambda self, path: "model",
    }
    del methods[missing_method]

    IncompleteTrainer = type("IncompleteTrainer", (Trainer,), methods)

    with pytest.raises(TypeError):
        IncompleteTrainer()  # type: ignore[abstract]


def test_minimal_concrete_subclass_satisfies_contract(tmp_path):
    class FakeTrainer(Trainer):
        def train(self, train_data):
            return {"fitted_on": train_data}

        def save(self, model, path):
            path.write_text(str(model))

        def load(self, path):
            return path.read_text()

    trainer = FakeTrainer()
    model = trainer.train(train_data="X,y")
    assert model == {"fitted_on": "X,y"}

    model_path = tmp_path / "model.txt"
    trainer.save(model, model_path)
    assert model_path.exists()

    loaded = trainer.load(model_path)
    assert loaded == str(model)
