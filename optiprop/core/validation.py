"""Structured validation results shared by the numerical core and user interfaces.

Validation is represented as data rather than immediately raising an exception.
This lets callers collect issues from several layers, display them in a Problems
panel, and only block execution when the final report contains errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Tuple


class Severity(str, Enum):
    """Severity level of a :class:`ValidationIssue`."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    """One immutable, UI-addressable validation finding.

    ``layer_id`` identifies an optical-system layer while ``parameter_path``
    identifies a field within that layer's property model.  Both are optional
    so the same type can represent project-wide or imported-field issues.

    ``details`` contains structured diagnostic context for logs and expanded UI
    views.  Mapping and collection values are recursively frozen so an issue
    cannot be changed indirectly after construction.
    """

    severity: Severity
    code: str
    message: str
    layer_id: str | None = None
    parameter_path: str | None = None
    suggested_fix: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            severity = (
                self.severity
                if isinstance(self.severity, Severity)
                else Severity(str(self.severity).lower())
            )
        except ValueError as exc:
            allowed = ", ".join(item.value for item in Severity)
            raise ValueError(f"severity must be one of: {allowed}.") from exc

        code = _required_text(self.code, "code")
        message = _required_text(self.message, "message")
        layer_id = _optional_text(self.layer_id, "layer_id")
        parameter_path = _optional_text(self.parameter_path, "parameter_path")
        suggested_fix = _optional_text(self.suggested_fix, "suggested_fix")
        details = _freeze_details(self.details)

        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "parameter_path", parameter_path)
        object.__setattr__(self, "suggested_fix", suggested_fix)
        object.__setattr__(self, "details", details)


@dataclass(frozen=True)
class ValidationReport:
    """An immutable collection of validation issues."""

    issues: Tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not all(isinstance(issue, ValidationIssue) for issue in issues):
            raise TypeError("issues must contain only ValidationIssue instances.")
        object.__setattr__(self, "issues", issues)

    def __iter__(self) -> Iterator[ValidationIssue]:
        return iter(self.issues)

    def __len__(self) -> int:
        return len(self.issues)

    @property
    def is_valid(self) -> bool:
        """Whether the report contains no errors."""

        return not self.errors

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        return self._of_severity(Severity.ERROR)

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        return self._of_severity(Severity.WARNING)

    @property
    def info(self) -> Tuple[ValidationIssue, ...]:
        return self._of_severity(Severity.INFO)

    def merge(self, *others: "ValidationReport") -> "ValidationReport":
        """Return a report containing this report followed by ``others``."""

        merged = list(self.issues)
        for other in others:
            if not isinstance(other, ValidationReport):
                raise TypeError("merge arguments must be ValidationReport instances.")
            merged.extend(other.issues)
        return ValidationReport(tuple(merged))

    def with_issue(self, issue: ValidationIssue) -> "ValidationReport":
        """Return a new report with ``issue`` appended."""

        if not isinstance(issue, ValidationIssue):
            raise TypeError("issue must be a ValidationIssue instance.")
        return ValidationReport(self.issues + (issue,))

    def raise_for_errors(self) -> "ValidationReport":
        """Raise :class:`ValidationError` if this report contains errors."""

        if not self.is_valid:
            raise ValidationError(self)
        return self

    def _of_severity(self, severity: Severity) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is severity)


class ValidationError(Exception):
    """Raised when execution is blocked by a validation report.

    The complete ``report`` remains attached so a GUI can show warnings and
    informational findings alongside the blocking ``errors``.
    """

    def __init__(self, report: ValidationReport) -> None:
        if not isinstance(report, ValidationReport):
            raise TypeError("report must be a ValidationReport instance.")
        if report.is_valid:
            raise ValueError("ValidationError requires a report containing errors.")

        self.report = report
        self.errors = report.errors
        super().__init__(_format_error_message(self.errors))


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _freeze_details(details: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(details, Mapping):
        raise TypeError("details must be a mapping.")
    if not all(isinstance(key, str) for key in details):
        raise TypeError("details keys must be strings.")
    return MappingProxyType(
        {key: _freeze_detail_value(value) for key, value in details.items()}
    )


def _freeze_detail_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("nested details mapping keys must be strings.")
        return MappingProxyType(
            {key: _freeze_detail_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_detail_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_detail_value(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _format_error_message(errors: Iterable[ValidationIssue]) -> str:
    errors = tuple(errors)
    if len(errors) == 1:
        issue = errors[0]
        location = []
        if issue.layer_id is not None:
            location.append(f"layer={issue.layer_id}")
        if issue.parameter_path is not None:
            location.append(f"parameter={issue.parameter_path}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"Validation failed: [{issue.code}] {issue.message}{suffix}"
    return f"Validation failed with {len(errors)} errors."


__all__ = [
    "Severity",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
]
