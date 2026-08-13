"""Abstract base class for data validation.

Validation checks that ingested data meets whatever assumptions the
rest of the pipeline (Transformer, Trainer) is going to rely on, and
records a pass/fail status somewhere durable (a status file today,
possibly an MLflow tag later) so a failed validation can halt the
pipeline before wasting time downstream.

wine_quality (current, tabular regression)
    validate(data) -> checks the CSV's columns/dtypes against
        schema.yaml, returns ValidationResult(is_valid, errors)
    Subclass: WineQualityValidator(Validator[pd.DataFrame])

Sketch — 05_defect_detection (CV, PyTorch, MVTec AD)
    validate(data) -> no "columns" exist here; checks would instead be:
        every image file opens/decodes, image dimensions are within an
        expected range, class folders (good/, broken_large/, ...) are
        non-empty
    Subclass: MVTecValidator(Validator[list[Path]])
        overrides validate() entirely with image-integrity checks; the
        *shape* of the contract (arbitrary data in, structured
        ValidationResult out, status persisted) is unchanged

Sketch — 06_rag_eval (LLM/RAG)
    validate(data) -> checks would be: documents are non-empty, text is
        decodable, no duplicate documents by content hash, document
        count is above a minimum useful-for-eval threshold
    Subclass: RagCorpusValidator(Validator[list[Document]])

Generalizes: keeping the contract to "take arbitrary data in, return a
structured ValidationResult, persist a status" — rather than baking in
"check columns against a schema dict" as the interface itself — is what
makes this reusable. Column/dtype schema-checking is one
*implementation* (wine_quality's), not part of the base contract.

Flag: none for this class as designed — see the generalization note
above on why schema-specific coupling was deliberately kept out of the
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, List, TypeVar

T = TypeVar("T")


@dataclass
class ValidationResult:
    """Structured outcome of a validation run.

    Attributes:
        is_valid: overall pass/fail for this validation run.
        errors: human-readable descriptions of every failed check.
            Empty when `is_valid` is True.
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)


class Validator(ABC, Generic[T]):
    """Abstract contract for validating ingested data before it's used.

    Subclasses implement `validate()` with whatever domain-appropriate
    checks make sense (schema/dtype checks for tabular data, file
    integrity checks for images, non-empty/dedup checks for documents).
    `run()` wraps `validate()` and persists the status via
    `write_status()`, mirroring wine_quality's existing status-file
    pattern while keeping the persistence mechanism swappable.
    """

    def __init__(self, status_file: Path) -> None:
        self.status_file = status_file

    @abstractmethod
    def validate(self, data: T) -> ValidationResult:
        """Run all checks against `data` and return a structured result."""
        raise NotImplementedError

    def write_status(self, result: ValidationResult) -> None:
        """Persist the validation status. Default: a plain-text status file.

        Subclasses may override for a different persistence mechanism
        (e.g. an MLflow tag) without changing the `validate()` contract.
        """
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            f.write(f"Validation status: {result.is_valid}")
            if result.errors:
                f.write("\n" + "\n".join(result.errors))

    def run(self, data: T) -> ValidationResult:
        """Validate `data` and persist the status. Convenience entry point."""
        result = self.validate(data)
        self.write_status(result)
        return result
