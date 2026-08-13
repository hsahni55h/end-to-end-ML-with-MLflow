"""Abstract base class for data ingestion.

Ingestion is the "get raw data from wherever it lives to somewhere the
rest of the pipeline can read it" stage. It deliberately says nothing
about *what* the ingested data looks like (a DataFrame, a list of image
paths, a folder of parsed documents) — that's why `load()` is generic
over the return type `T`.

wine_quality (current, tabular regression)
    fetch() -> download the winequality zip from a URL, save it locally
    load()  -> read the extracted CSV into a pandas.DataFrame
    Subclass: WineQualityIngestor(Ingestor[pd.DataFrame])

Sketch — 05_defect_detection (CV, PyTorch, MVTec AD)
    fetch() -> download + extract the MVTec AD category archive
    load()  -> return e.g. a list[Path] of image files (or a thin
               dataclass wrapping the train/test image directories) —
               NOT a DataFrame. MVTec AD already ships split into
               good/ and defect-type subfolders on disk.
    Subclass: MVTecIngestor(Ingestor[list[Path]])

Sketch — 06_rag_eval (LLM/RAG)
    fetch() -> copy/download the source document set (PDFs, markdown)
               into a local corpus directory
    load()  -> return list[Document] (raw text + metadata) — also not
               a DataFrame.
    Subclass: RagCorpusIngestor(Ingestor[list[Document]])

Generalizes: the fetch()/load() split (remote-pull vs.
materialize-into-memory) holds for both sketches above — neither
produces a DataFrame, but neither needed the interface changed, only
`T` substituted.

Flag: none for this class as designed — the generic return type is
exactly what avoids a tabular-only interface here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Ingestor(ABC, Generic[T]):
    """Abstract contract for pulling raw data into a locally usable form.

    Subclasses implement two stages:
      1. `fetch()` — pull raw bytes from wherever the source data lives
         (a URL, an API, a database, a shared filesystem) into local
         storage. Side-effecting; returns nothing.
      2. `load()` — read what `fetch()` produced into a form the rest
         of the pipeline (Validator, Transformer, ...) can consume.
         The returned type is deliberately generic (`T`) — a
         pandas.DataFrame for tabular projects, a list of file paths
         for images, a list of parsed documents for RAG, etc.

    `ingest()` is a concrete convenience that runs both stages;
    subclasses normally only override `fetch()` and `load()`.
    """

    @abstractmethod
    def fetch(self) -> None:
        """Pull raw data from its source into local storage.

        Must be idempotent: calling this when the data already exists
        locally should not error or re-download unnecessarily (see
        wine_quality's existing `download_file` for the pattern this
        formalizes).
        """
        raise NotImplementedError

    @abstractmethod
    def load(self) -> T:
        """Load the locally materialized raw data into memory.

        Returns:
            T: project-specific in-memory representation of the
                ingested data (e.g. pandas.DataFrame, list[Path],
                list[Document]).
        """
        raise NotImplementedError

    def ingest(self) -> T:
        """Run `fetch()` followed by `load()`. Convenience entry point."""
        self.fetch()
        return self.load()
