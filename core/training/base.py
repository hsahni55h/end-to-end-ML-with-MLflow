"""Abstract base class for model training.

Training fits a model/artifact against prepared data and persists it.
The interface is intentionally a single `train()` entry point plus a
`save()`/`load()` pair — it does not assume a scikit-learn-style single
`.fit()` call versus a PyTorch-style multi-epoch loop; that distinction
lives inside the implementation, not the interface.

wine_quality (current, tabular regression)
    train(data)      -> ElasticNet().fit(X, y) in one call
    save(model, path) -> joblib.dump(model, path)
    Subclass: ElasticNetTrainer(Trainer[ElasticNet])

Sketch — 05_defect_detection (CV, PyTorch, MVTec AD)
    train(data)      -> a multi-epoch training loop internally (forward/
        backward pass, optimizer step, GPU/CPU device handling); still
        a single `train()` call from the outside
    save(model, path) -> torch.save(model.state_dict(), path)
    Subclass: DefectDetectionTrainer(Trainer[torch.nn.Module])

Sketch — 06_rag_eval (LLM/RAG)
    This is the shakiest fit of the five base classes. There is no
    "training" in the classical sense for a RAG pipeline built on a
    frozen, off-the-shelf LLM + embedding model — the closest analogue
    is "build/persist the vector index" (embed chunks, upsert into
    Chroma/Qdrant).
    Tentative subclass: RagIndexBuilder(Trainer[VectorStoreHandle])
        train(data)      -> embed + upsert chunks into the vector
                             store; the "model" returned is a
                             handle/client to that store
        save(model, path) -> persist the store's on-disk location /
                             connection details
    This mapping is a stretch and is called out explicitly below rather
    than treated as settled.

Flag: mapping 06_rag_eval onto `Trainer` at all is an open question.
Building a vector index isn't "fitting a model" the way the other four
projects mean it, and forcing it through this interface purely for
framework consistency may cost more clarity than it buys. Two options
to weigh before Phase 4 (S4.4):
  (a) keep it — treat "index build" as this domain's `train()`, accept
      the metaphor stretch for framework consistency; or
  (b) introduce a separate `Indexer` concept for 06_rag_eval instead of
      forcing it through `Trainer`.
Left open here per this session's scope (interfaces + sketch only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")
ModelT = TypeVar("ModelT")


class Trainer(ABC, Generic[T, ModelT]):
    """Abstract contract for fitting and persisting a model/artifact.

    Subclasses implement `train()`, `save()`, and `load()`. `train()` is
    the single entry point regardless of whether fitting is a one-shot
    call (scikit-learn) or an iterative loop (PyTorch) — that
    distinction lives inside the implementation, not the interface.
    """

    @abstractmethod
    def train(self, train_data: T) -> ModelT:
        """Fit a model/artifact against `train_data` and return it."""
        raise NotImplementedError

    @abstractmethod
    def save(self, model: ModelT, path: Path) -> None:
        """Persist `model` to `path`."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> ModelT:
        """Load a previously-saved model/artifact from `path`."""
        raise NotImplementedError
