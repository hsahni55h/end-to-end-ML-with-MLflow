"""wine_quality Validator: checks the ingested CSV against schema.yaml.

Subclasses `core.validation.base.Validator[pd.DataFrame]`. The base class
already provides `write_status()`/`run()` (status-file persistence); only
the schema/dtype checks themselves are domain-specific here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from core.validation.base import ValidationResult, Validator


class WineQualityValidator(Validator[pd.DataFrame]):
    """Validates that every column in the data matches schema.yaml's dtype."""

    def __init__(self, status_file: Path, schema: Dict[str, str]) -> None:
        super().__init__(status_file)
        self.schema = schema

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        errors = []
        for column in data.columns:
            if column not in self.schema:
                errors.append(f"Column {column} is not in the schema")
                continue

            expected_dtype = self.schema[column]
            actual_dtype = str(data[column].dtype)
            if actual_dtype != expected_dtype:
                errors.append(
                    f"Data type for column {column} does not match. "
                    f"Expected: {expected_dtype}, Found: {actual_dtype}"
                )

        return ValidationResult(is_valid=not errors, errors=errors)
