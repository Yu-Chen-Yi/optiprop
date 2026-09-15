"""Contract tests for structured, immutable validation reports."""

from dataclasses import FrozenInstanceError

import pytest

from optiprop.core.validation import (
    Severity,
    ValidationError,
    ValidationIssue,
    ValidationReport,
)


def _issue(
    code: str,
    severity: Severity,
    *,
    message: str | None = None,
    layer_id: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message or code,
        parameter_path="distance_m",
        layer_id=layer_id,
        details={"value": 1.0},
    )


def test_severity_is_a_stable_string_enum() -> None:
    assert Severity.INFO.value == "info"
    assert Severity.WARNING.value == "warning"
    assert Severity.ERROR.value == "error"


def test_validation_issue_is_frozen_and_snapshots_details() -> None:
    details = {"distance_m": 1.0}
    issue = ValidationIssue(
        code="sampling.example",
        severity=Severity.WARNING,
        message="Example warning",
        parameter_path="distance_m",
        layer_id="layer-7",
        details=details,
    )
    details["distance_m"] = 2.0

    assert issue.code == "sampling.example"
    assert issue.severity is Severity.WARNING
    assert issue.message == "Example warning"
    assert issue.parameter_path == "distance_m"
    assert issue.layer_id == "layer-7"
    assert issue.details["distance_m"] == 1.0
    with pytest.raises(FrozenInstanceError):
        issue.code = "changed"
    with pytest.raises(TypeError):
        issue.details["distance_m"] = 3.0


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("code", "", id="empty-code"),
        pytest.param("message", "", id="empty-message"),
        pytest.param("details", [], id="details-must-be-mapping"),
    ],
)
def test_invalid_issue_inputs_are_rejected(keyword: str, value) -> None:
    parameters = {
        "code": "sampling.valid_code",
        "severity": Severity.WARNING,
        "message": "A useful message",
        "details": {},
    }
    parameters[keyword] = value

    with pytest.raises((TypeError, ValueError)):
        ValidationIssue(**parameters)


def test_string_severity_is_canonicalized_to_enum() -> None:
    issue = ValidationIssue(
        code="sampling.example",
        severity="WARNING",
        message="Example warning",
    )

    assert issue.severity is Severity.WARNING


def test_report_is_immutable_and_filters_issues_by_severity() -> None:
    info = _issue("sampling.info", Severity.INFO)
    warning = _issue("sampling.warning", Severity.WARNING)
    error = _issue("sampling.error", Severity.ERROR)
    source = [info, warning, error]

    report = ValidationReport(issues=source)
    source.clear()

    assert report.issues == (info, warning, error)
    assert report.info == (info,)
    assert report.warnings == (warning,)
    assert report.errors == (error,)
    assert not report.is_valid
    with pytest.raises(FrozenInstanceError):
        report.issues = ()
    with pytest.raises(AttributeError):
        report.issues.append(info)


def test_empty_report_is_valid_and_raise_is_a_noop() -> None:
    report = ValidationReport()

    assert report.issues == ()
    assert report.is_valid
    assert report.warnings == ()
    assert report.errors == ()
    assert report.raise_for_errors() is report


def test_merge_returns_new_report_preserves_order_and_does_not_mutate_inputs() -> None:
    first_issue = _issue("sampling.first", Severity.INFO)
    second_issue = _issue("sampling.second", Severity.WARNING)
    first = ValidationReport((first_issue,))
    second = ValidationReport((second_issue,))

    merged = first.merge(second)

    assert merged is not first
    assert merged is not second
    assert merged.issues == (first_issue, second_issue)
    assert first.issues == (first_issue,)
    assert second.issues == (second_issue,)


def test_raise_if_errors_raises_structured_validation_error() -> None:
    warning = _issue("sampling.warning", Severity.WARNING)
    error = _issue(
        "sampling.invalid_output_grid",
        Severity.ERROR,
        message="Output grid is invalid",
        layer_id="propagation-3",
    )
    report = ValidationReport((warning, error))

    with pytest.raises(ValidationError) as caught:
        report.raise_for_errors()

    assert caught.value.report is report
    assert caught.value.errors == (error,)
    assert "sampling.invalid_output_grid" in str(caught.value)
    assert "Output grid is invalid" in str(caught.value)


def test_warning_only_report_does_not_raise() -> None:
    report = ValidationReport(
        (_issue("sampling.wraparound_risk", Severity.WARNING),)
    )

    assert report.is_valid
    assert report.raise_for_errors() is report
