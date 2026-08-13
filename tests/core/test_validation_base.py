"""Interface-contract tests for `core.validation.base.Validator`.

Confirms the ABC can't be instantiated directly, a minimal concrete
subclass must implement `validate()`, and `run()` correctly wires
`validate()` + `write_status()` (default status-file persistence).
"""

from __future__ import annotations

import pytest

from core.validation.base import ValidationResult, Validator


def test_cannot_instantiate_abstract_validator(tmp_path):
    with pytest.raises(TypeError):
        Validator(tmp_path / "status.txt")  # type: ignore[abstract]


def test_subclass_missing_validate_cannot_instantiate(tmp_path):
    class MissingValidate(Validator):
        pass

    with pytest.raises(TypeError):
        MissingValidate(tmp_path / "status.txt")  # type: ignore[abstract]


def test_validation_result_defaults_to_empty_errors():
    result = ValidationResult(is_valid=True)
    assert result.errors == []


def test_run_writes_status_file_on_success(tmp_path):
    status_file = tmp_path / "nested" / "status.txt"

    class PassingValidator(Validator):
        def validate(self, data):
            return ValidationResult(is_valid=True)

    validator = PassingValidator(status_file)
    result = validator.run(data=None)

    assert result.is_valid is True
    assert status_file.exists()
    assert "Validation status: True" in status_file.read_text()


def test_run_writes_errors_on_failure(tmp_path):
    status_file = tmp_path / "status.txt"

    class FailingValidator(Validator):
        def validate(self, data):
            return ValidationResult(is_valid=False, errors=["column X missing"])

    validator = FailingValidator(status_file)
    result = validator.run(data=None)

    assert result.is_valid is False
    content = status_file.read_text()
    assert "Validation status: False" in content
    assert "column X missing" in content
