"""Immutable execution results for an optical system."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Tuple
from uuid import UUID

from ..core import Field2D, ValidationReport
from ..propagation import PropagationCancelled, PropagationResult


class LayerExecutionStatus(str, Enum):
    """Terminal state recorded for a layer visited by a system run."""

    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class LayerResult:
    """One layer's immutable field snapshot and execution provenance."""

    layer_id: UUID
    layer_name: str
    layer_type: str
    status: LayerExecutionStatus
    input_field: Field2D
    output_field: Field2D
    validation: ValidationReport = field(default_factory=ValidationReport)
    elapsed_s: float = 0.0
    parameter_hash: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    propagation_result: PropagationResult | None = None
    cache_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_id", _uuid(self.layer_id, "layer_id"))
        object.__setattr__(
            self, "layer_name", _required_text(self.layer_name, "layer_name")
        )
        object.__setattr__(
            self, "layer_type", _required_text(self.layer_type, "layer_type")
        )
        try:
            status = (
                self.status
                if isinstance(self.status, LayerExecutionStatus)
                else LayerExecutionStatus(self.status)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("status must be 'completed' or 'skipped'.") from exc
        if not isinstance(self.input_field, Field2D):
            raise TypeError("input_field must be a Field2D.")
        if not isinstance(self.output_field, Field2D):
            raise TypeError("output_field must be a Field2D.")
        if not isinstance(self.validation, ValidationReport):
            raise TypeError("validation must be a ValidationReport.")
        if self.propagation_result is not None and not isinstance(
            self.propagation_result, PropagationResult
        ):
            raise TypeError("propagation_result must be a PropagationResult or None.")
        elapsed_s = _nonnegative_finite(self.elapsed_s, "elapsed_s")
        parameter_hash = _required_text(self.parameter_hash, "parameter_hash")
        if self.cache_key is not None:
            cache_key = _required_text(self.cache_key, "cache_key")
        else:
            cache_key = None
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "elapsed_s", elapsed_s)
        object.__setattr__(self, "parameter_hash", parameter_hash)
        object.__setattr__(
            self, "diagnostics", _deep_freeze(self.diagnostics)
        )
        object.__setattr__(self, "cache_key", cache_key)

    @property
    def warnings(self):
        return self.validation.warnings


@dataclass(frozen=True)
class OpticalSystemResult:
    """Input, per-layer snapshots, and final field from one system run."""

    system_id: UUID
    input_field: Field2D
    final_field: Field2D
    layer_results: Tuple[LayerResult, ...] = ()
    validation: ValidationReport = field(default_factory=ValidationReport)
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_id", _uuid(self.system_id, "system_id"))
        if not isinstance(self.input_field, Field2D):
            raise TypeError("input_field must be a Field2D.")
        if not isinstance(self.final_field, Field2D):
            raise TypeError("final_field must be a Field2D.")
        results = tuple(self.layer_results)
        if not all(isinstance(result, LayerResult) for result in results):
            raise TypeError("layer_results must contain only LayerResult instances.")
        if not isinstance(self.validation, ValidationReport):
            raise TypeError("validation must be a ValidationReport.")
        object.__setattr__(self, "layer_results", results)
        object.__setattr__(
            self, "elapsed_s", _nonnegative_finite(self.elapsed_s, "elapsed_s")
        )

    @property
    def output_field(self) -> Field2D:
        """Alias used by single-operation consumers."""

        return self.final_field

    @property
    def snapshots(self) -> Tuple[Field2D, ...]:
        """Input followed by the output of every visited layer."""

        return (self.input_field,) + tuple(
            result.output_field for result in self.layer_results
        )

    @property
    def completed_layer_ids(self) -> Tuple[UUID, ...]:
        return tuple(
            result.layer_id
            for result in self.layer_results
            if result.status is LayerExecutionStatus.COMPLETED
        )

    def result_for(self, layer_id: UUID | str) -> LayerResult:
        normalized = _uuid(layer_id, "layer_id")
        for result in self.layer_results:
            if result.layer_id == normalized:
                return result
        raise KeyError(f"No result exists for layer {normalized}.")


class SystemExecutionCancelled(PropagationCancelled):
    """Cancellation carrying all layer results completed before cancellation."""

    def __init__(
        self,
        partial_result: OpticalSystemResult,
        reason: str | None = None,
    ) -> None:
        if not isinstance(partial_result, OpticalSystemResult):
            raise TypeError("partial_result must be an OpticalSystemResult.")
        self.partial_result = partial_result
        super().__init__(reason)


class SystemExecutionFailed(RuntimeError):
    """Unexpected layer failure carrying completed partial results."""

    def __init__(
        self,
        layer_id: UUID,
        partial_result: OpticalSystemResult,
        cause: BaseException,
    ) -> None:
        self.layer_id = _uuid(layer_id, "layer_id")
        if not isinstance(partial_result, OpticalSystemResult):
            raise TypeError("partial_result must be an OpticalSystemResult.")
        if not isinstance(cause, BaseException):
            raise TypeError("cause must be an exception.")
        self.partial_result = partial_result
        self.cause = cause
        super().__init__(f"Layer {self.layer_id} execution failed: {cause}")


def _uuid(value: UUID | str, name: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be a UUID or UUID string.") from exc


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return normalized


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("diagnostics keys must be strings.")
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


__all__ = [
    "LayerExecutionStatus",
    "LayerResult",
    "OpticalSystemResult",
    "SystemExecutionCancelled",
    "SystemExecutionFailed",
]
