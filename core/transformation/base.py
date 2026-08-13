"""Abstract base class for data transformation.

Transformation turns validated raw data into whatever a Trainer expects
to fit against: feature engineering, encoding, splitting into
train/test, tokenization/chunking, image augmentation, and so on.

wine_quality (current, tabular regression)
    transform(data) -> currently just the 75/25 train_test_split; real
        feature engineering (scaling, encoding, outlier handling) is
        S1.2 scope, not this base class
    Subclass: WineQualityTransformer(Transformer[pd.DataFrame])

Sketch — 05_defect_detection (CV, PyTorch, MVTec AD)
    transform(data) -> resize/normalize images, apply augmentations
        (random crop/flip), return torch Dataset/DataLoader-ready
        tensors. MVTec AD ships with its own train/test folder split,
        so `split()` becomes a no-op passthrough rather than a random
        split — see the flag below.
    Subclass: MVTecTransformer(Transformer[torch.utils.data.Dataset])

Sketch — 06_rag_eval (LLM/RAG)
    transform(data) -> chunk documents into passages, generate
        embeddings, return vectors ready for indexing. "Train/test
        split" doesn't map onto a document corpus the same way — it's
        closer to "corpus vs. a hand-curated held-out eval question
        set", which are two independently-built inputs, not a split of
        one dataset.
    Subclass: RagTransformer(Transformer[list[EmbeddedChunk]])

Generalizes: the fit/transform separation (learn transformation
parameters from data, then apply them) holds across all three — scaler
statistics, augmentation config, and an embedding model are all "fit
once, apply repeatedly" in the same shape.

Flag: `split()` is provided as a concrete default (sklearn-style random
split) because wine_quality/fraud_graph/credit_default/energy_forecast
all need some form of train/test partitioning. It does NOT generalize
cleanly to 05_defect_detection (pre-defined folder split) or
06_rag_eval (corpus vs. eval-set are different inputs, not a split of
one). Kept concrete rather than abstract specifically so those two
projects can override it to a passthrough/no-op instead of being forced
to implement a meaningless random split. Revisit if this override
pattern feels awkward once 05/06 are actually built — an alternative
would be to drop `split()` from the base entirely and make it
project-specific from the start.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

from sklearn.model_selection import train_test_split

T = TypeVar("T")


class Transformer(ABC, Generic[T]):
    """Abstract contract for transforming validated data into model-ready form.

    Subclasses implement `fit()` and `transform()`. `split()` has a
    concrete tabular-appropriate default (random train/test split) that
    non-tabular projects are expected to override — see the module
    docstring's flagged generalization gap.
    """

    @abstractmethod
    def fit(self, data: T) -> "Transformer[T]":
        """Learn any transformation parameters from `data` (e.g. scaler
        statistics, an embedding model, augmentation config).

        Returns:
            Transformer[T]: self, to allow chaining (`fit(data).transform(data)`).
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: T) -> T:
        """Apply the fitted transformation to `data` and return the result."""
        raise NotImplementedError

    def fit_transform(self, data: T) -> T:
        """Convenience: `fit(data)` then `transform(data)`."""
        self.fit(data)
        return self.transform(data)

    def split(
        self, data: T, test_size: float = 0.25, random_state: int = 42
    ) -> Tuple[T, T]:
        """Split `data` into train/test partitions.

        Concrete default assumes `data` is a pandas.DataFrame (or a
        similarly indexable structure) and performs a random split via
        scikit-learn. Override for projects with a pre-defined split
        (e.g. image datasets with fixed train/test folders) or where
        train/eval come from independently-curated sources (e.g. a RAG
        corpus vs. a hand-written eval set) — see the module docstring's
        flagged generalization gap.
        """
        return train_test_split(data, test_size=test_size, random_state=random_state)
