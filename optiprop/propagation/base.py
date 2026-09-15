"""Shared propagation contracts for the OptiProp 2.0 numerical core.

This module intentionally contains no diffraction algorithm.  It defines the
immutable request/result objects and cooperative cancellation primitive used
by numerical implementations, services, and the GUI.
"""

from __future__ import annotations

import math
import numbers
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple

from ..core import Field2D, Grid2D, ValidationReport

if TYPE_CHECKING:
    from .sampling import SamplingReport


class PropagationMethod(str, Enum):
    """Numerical propagation method selected for a request."""

    ASM = "asm"
    BLAS = "blas"
    FRESNEL_TF = "fresnel_tf"
    FRESNEL_SCALED = "fresnel_scaled"
    RS_FFT = "rs_fft"
    RS_DIRECT = "rs_direct"


class EvanescentPolicy(str, Enum):
    """Treatment of non-propagating angular-spectrum components."""

    DISCARD = "discard"
    DECAY = "decay"
    KEEP = "keep"


class PaddingMode(str, Enum):
    """How a propagator constructs its computational grid."""

    NONE = "none"
    AUTO = "auto"
    FACTOR = "factor"
    EXPLICIT = "explicit"


class PrecisionPolicy(str, Enum):
    """Complex precision used by a propagation request."""

    INHERIT = "inherit"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


@dataclass(frozen=True)
class PaddingSpec:
    """Immutable zero-padding request.

    ``shape`` is always canonical ``(ny, nx)``.  Its relationship to an input
    grid is checked later by a concrete propagator because this base contract
    has no input field at construction time.
    """

    mode: PaddingMode = PaddingMode.AUTO
    factor: float | None = None
    shape: Tuple[int, int] | None = None

    def __post_init__(self) -> None:
        mode = _coerce_enum(self.mode, PaddingMode, "mode")
        object.__setattr__(self, "mode", mode)

        if mode in (PaddingMode.NONE, PaddingMode.AUTO):
            if self.factor is not None or self.shape is not None:
                raise ValueError(
                    f"Padding mode {mode.value!r} does not accept factor or shape."
                )
            return

        if mode is PaddingMode.FACTOR:
            if self.shape is not None:
                raise ValueError("Factor padding does not accept an explicit shape.")
            factor = _finite_float(self.factor, "factor")
            if factor < 1.0:
                raise ValueError("Padding factor must be at least 1.")
            object.__setattr__(self, "factor", factor)
            return

        if self.factor is not None:
            raise ValueError("Explicit padding does not accept a factor.")
        shape = _validated_shape(self.shape)
        object.__setattr__(self, "shape", shape)

    @classmethod
    def none(cls) -> "PaddingSpec":
        return cls(mode=PaddingMode.NONE)

    @classmethod
    def auto(cls) -> "PaddingSpec":
        return cls(mode=PaddingMode.AUTO)

    @classmethod
    def by_factor(cls, factor: float) -> "PaddingSpec":
        return cls(mode=PaddingMode.FACTOR, factor=factor)

    @classmethod
    def explicit(cls, shape: Sequence[int]) -> "PaddingSpec":
        return cls(mode=PaddingMode.EXPLICIT, shape=tuple(shape))


@dataclass(frozen=True)
class PropagationSpec:
    """Immutable specification for one propagation operation.

    ``medium_index=None`` means use ``field.medium_index``.  ``output_grid`` is
    optional because same-grid algorithms do not require it; algorithms that
    support scaled or arbitrary output sampling must validate it explicitly.
    Method-specific, serializable settings can be carried in ``options``.
    """

    method: PropagationMethod
    distance_m: float
    medium_index: complex | None = None
    padding: PaddingSpec = field(default_factory=PaddingSpec)
    evanescent_policy: EvanescentPolicy = EvanescentPolicy.DISCARD
    precision: PrecisionPolicy = PrecisionPolicy.INHERIT
    output_grid: Grid2D | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method",
            _coerce_enum(self.method, PropagationMethod, "method"),
        )
        object.__setattr__(
            self, "distance_m", _finite_float(self.distance_m, "distance_m")
        )
        if self.medium_index is not None:
            object.__setattr__(
                self,
                "medium_index",
                _validated_medium_index(self.medium_index),
            )
        if not isinstance(self.padding, PaddingSpec):
            raise TypeError("padding must be a PaddingSpec.")
        object.__setattr__(
            self,
            "evanescent_policy",
            _coerce_enum(
                self.evanescent_policy,
                EvanescentPolicy,
                "evanescent_policy",
            ),
        )
        object.__setattr__(
            self,
            "precision",
            _coerce_enum(self.precision, PrecisionPolicy, "precision"),
        )
        if self.output_grid is not None and not isinstance(
            self.output_grid, Grid2D
        ):
            raise TypeError("output_grid must be a Grid2D or None.")
        object.__setattr__(self, "options", _snapshot_mapping(self.options, "options"))

    def resolved_medium_index(self, field: Field2D) -> complex:
        """Return the explicit medium index or the input field's current index."""

        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        return field.medium_index if self.medium_index is None else self.medium_index


@dataclass(frozen=True)
class PropagationResult:
    """Result and structured run context for one propagation operation."""

    field: Field2D
    spec: PropagationSpec
    elapsed_s: float = 0.0
    sampling: "SamplingReport | None" = None
    validation: ValidationReport = field(default_factory=ValidationReport)
    power_before: float | None = None
    power_after: float | None = None
    cache_key: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.field, Field2D):
            raise TypeError("field must be a Field2D.")
        if not isinstance(self.spec, PropagationSpec):
            raise TypeError("spec must be a PropagationSpec.")
        elapsed_s = _finite_float(self.elapsed_s, "elapsed_s")
        if elapsed_s < 0.0:
            raise ValueError("elapsed_s must be non-negative.")
        if self.sampling is not None:
            from .sampling import SamplingReport

            if not isinstance(self.sampling, SamplingReport):
                raise TypeError("sampling must be a SamplingReport or None.")
        if not isinstance(self.validation, ValidationReport):
            raise TypeError("validation must be a ValidationReport.")
        power_before = _optional_nonnegative_float(
            self.power_before, "power_before"
        )
        power_after = _optional_nonnegative_float(self.power_after, "power_after")
        cache_key = _optional_text(self.cache_key, "cache_key")
        object.__setattr__(self, "elapsed_s", elapsed_s)
        object.__setattr__(self, "power_before", power_before)
        object.__setattr__(self, "power_after", power_after)
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(
            self,
            "diagnostics",
            _snapshot_mapping(self.diagnostics, "diagnostics"),
        )

    @property
    def warnings(self):
        """Structured warnings associated with this result."""

        return self.validation.warnings


class PropagationCancelled(RuntimeError):
    """Raised when cooperative propagation cancellation is observed."""

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        message = "Propagation was cancelled."
        if reason:
            message = f"{message} {reason}"
        super().__init__(message)


class CancellationToken:
    """A thread-safe, one-shot cooperative cancellation token.

    Calling :meth:`cancel` is safe from any thread.  Long-running algorithms
    should call :meth:`throw_if_cancelled` between layers, slices, or direct
    integration chunks.  A single device FFT generally cannot be interrupted
    safely in the middle of the backend call.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        if reason is not None and not isinstance(reason, str):
            raise TypeError("Cancellation reason must be a string or None.")
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def cancelled(self) -> bool:
        """Alias for integrations that use ``token.cancelled``."""

        return self.is_cancelled

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancelled or timeout; return whether cancellation occurred."""

        return self._event.wait(timeout)

    def throw_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise PropagationCancelled(self.reason)


class Propagator(ABC):
    """Abstract interface implemented by numerical propagation backends."""

    def validate(
        self,
        field: Field2D,
        spec: PropagationSpec,
    ) -> ValidationReport:
        """Return structured backend validation findings."""

        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        if not isinstance(spec, PropagationSpec):
            raise TypeError("spec must be a PropagationSpec.")
        return ValidationReport()

    @abstractmethod
    def propagate(
        self,
        field: Field2D,
        spec: PropagationSpec,
        cancel_token: CancellationToken | None = None,
    ) -> PropagationResult:
        """Propagate ``field`` according to ``spec`` without mutating either."""

        raise NotImplementedError

    def __call__(
        self,
        field: Field2D,
        spec: PropagationSpec,
        cancel_token: CancellationToken | None = None,
    ) -> PropagationResult:
        return self.propagate(field, spec, cancel_token=cancel_token)


def _coerce_enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}.") from exc


def _finite_float(value: object, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _validated_shape(value: object) -> Tuple[int, int]:
    if value is None or isinstance(value, (str, bytes)):
        raise TypeError("Explicit padding shape must be a (ny, nx) sequence.")
    try:
        shape = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "Explicit padding shape must be a (ny, nx) sequence."
        ) from exc
    if len(shape) != 2:
        raise ValueError("Explicit padding shape must contain exactly (ny, nx).")
    if any(isinstance(size, bool) or not isinstance(size, numbers.Integral) for size in shape):
        raise TypeError("Explicit padding ny and nx must be integers.")
    normalized = int(shape[0]), int(shape[1])
    if normalized[0] <= 0 or normalized[1] <= 0:
        raise ValueError("Explicit padding ny and nx must be positive.")
    return normalized


def _validated_medium_index(value: object) -> complex:
    if isinstance(value, bool):
        raise TypeError("medium_index must be a real or complex number.")
    try:
        normalized = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("medium_index must be a real or complex number.") from exc
    if not (
        math.isfinite(normalized.real) and math.isfinite(normalized.imag)
    ):
        raise ValueError("medium_index must be finite.")
    if normalized.real <= 0.0:
        raise ValueError("medium_index.real must be positive.")
    return normalized


def _optional_nonnegative_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    normalized = _finite_float(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _snapshot_mapping(
    value: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    snapshot = dict(value)
    if not all(isinstance(key, str) for key in snapshot):
        raise TypeError(f"{name} keys must be strings.")
    return MappingProxyType(
        {key: _freeze_option_value(item) for key, item in snapshot.items()}
    )


def _freeze_option_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Nested option mapping keys must be strings.")
        return MappingProxyType(
            {key: _freeze_option_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_option_value(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


__all__ = [
    "CancellationToken",
    "EvanescentPolicy",
    "PaddingMode",
    "PaddingSpec",
    "PrecisionPolicy",
    "PropagationCancelled",
    "PropagationMethod",
    "PropagationResult",
    "PropagationSpec",
    "Propagator",
]
