"""Immutable inspection, mapping, and report models for field I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ..core import EX_EY_COMPONENTS, SCALAR_COMPONENTS, Field2D, ValidationIssue


class FieldFormat(str, Enum):
    NPZ = "npz"
    MAT = "mat"
    ZBF = "zbf"


class MappingConfidence(str, Enum):
    EXACT = "exact"
    SUGGESTED = "suggested"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


class LengthUnit(str, Enum):
    M = "m"
    MM = "mm"
    UM = "um"
    NM = "nm"

    @property
    def scale_to_m(self) -> float:
        return {
            LengthUnit.M: 1.0,
            LengthUnit.MM: 1e-3,
            LengthUnit.UM: 1e-6,
            LengthUnit.NM: 1e-9,
        }[self]


class AxisOrder(str, Enum):
    YX = "yx"
    XY = "xy"


class PhaseUnit(str, Enum):
    RAD = "rad"
    DEG = "deg"


class AmplitudeConvention(str, Enum):
    AMPLITUDE = "amplitude"
    INTENSITY = "intensity"


@dataclass(frozen=True)
class ImportMapping:
    """Explicit mapping for a noncanonical dataset.

    Units are never inferred from values.  A key-based spatial or wavelength
    mapping therefore needs its corresponding explicit unit unless a direct
    SI override is supplied.
    """

    data_keys: tuple[str, ...] = ()
    amplitude_keys: tuple[str, ...] = ()
    phase_keys: tuple[str, ...] = ()
    components: tuple[str, ...] = SCALAR_COMPONENTS
    dx_key: str | None = None
    dy_key: str | None = None
    pixel_size_key: str | None = None
    wavelength_key: str | None = None
    medium_index_real_key: str | None = None
    medium_index_imag_key: str | None = None
    x_center_key: str | None = None
    y_center_key: str | None = None
    z_key: str | None = None
    nx_key: str | None = None
    ny_key: str | None = None
    metadata_key: str | None = None
    dx_m: float | None = None
    dy_m: float | None = None
    wavelength_m: float | None = None
    # Air is an explicit, persisted mapping default for the UI.  Import code
    # never derives refractive index from filename or numeric magnitude.
    medium_index: complex | None = 1.0
    x_center_m: float = 0.0
    y_center_m: float = 0.0
    z_m: float = 0.0
    spatial_unit: LengthUnit | None = None
    wavelength_unit: LengthUnit | None = None
    axis_order: AxisOrder = AxisOrder.YX
    flip_x: bool = False
    flip_y: bool = False
    phase_unit: PhaseUnit = PhaseUnit.RAD
    amplitude_convention: AmplitudeConvention = AmplitudeConvention.AMPLITUDE
    phase_sign: int = 1
    conjugate: bool = False

    def __post_init__(self) -> None:
        data_keys = _key_tuple(self.data_keys, "data_keys")
        amplitude_keys = _key_tuple(self.amplitude_keys, "amplitude_keys")
        phase_keys = _key_tuple(self.phase_keys, "phase_keys")
        if data_keys and (amplitude_keys or phase_keys):
            raise ValueError(
                "Use either data_keys or amplitude_keys/phase_keys, not both."
            )
        if amplitude_keys and len(amplitude_keys) != len(phase_keys):
            raise ValueError(
                "amplitude_keys and phase_keys must contain the same count."
            )
        if phase_keys and not amplitude_keys:
            raise ValueError("phase_keys require amplitude_keys.")
        if len(data_keys) > 2 or len(amplitude_keys) > 2:
            raise ValueError("A field mapping supports at most two components.")
        components = tuple(self.components)
        if components not in (SCALAR_COMPONENTS, EX_EY_COMPONENTS):
            raise ValueError(
                "components must be ('scalar',) or ('Ex', 'Ey')."
            )
        if len(data_keys) == 2 and components != EX_EY_COMPONENTS:
            raise ValueError("Two data_keys require components ('Ex', 'Ey').")
        if len(amplitude_keys) == 2 and components != EX_EY_COMPONENTS:
            raise ValueError(
                "Two amplitude/phase pairs require components ('Ex', 'Ey')."
            )

        optional_keys = (
            "dx_key",
            "dy_key",
            "pixel_size_key",
            "wavelength_key",
            "medium_index_real_key",
            "medium_index_imag_key",
            "x_center_key",
            "y_center_key",
            "z_key",
            "nx_key",
            "ny_key",
            "metadata_key",
        )
        for name in optional_keys:
            object.__setattr__(self, name, _optional_key(getattr(self, name), name))
        if self.pixel_size_key and (self.dx_key or self.dy_key):
            raise ValueError(
                "pixel_size_key cannot be combined with dx_key or dy_key."
            )
        object.__setattr__(
            self, "spatial_unit", _optional_enum(self.spatial_unit, LengthUnit)
        )
        object.__setattr__(
            self,
            "wavelength_unit",
            _optional_enum(self.wavelength_unit, LengthUnit),
        )
        object.__setattr__(self, "axis_order", _enum(self.axis_order, AxisOrder))
        object.__setattr__(self, "phase_unit", _enum(self.phase_unit, PhaseUnit))
        object.__setattr__(
            self,
            "amplitude_convention",
            _enum(self.amplitude_convention, AmplitudeConvention),
        )
        for name in ("flip_x", "flip_y", "conjugate"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        if isinstance(self.phase_sign, bool) or self.phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be either -1 or 1.")
        object.__setattr__(self, "data_keys", data_keys)
        object.__setattr__(self, "amplitude_keys", amplitude_keys)
        object.__setattr__(self, "phase_keys", phase_keys)
        object.__setattr__(self, "components", components)
        for name in ("dx_m", "dy_m", "wavelength_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive_float(value, name))
        if self.medium_index is not None:
            object.__setattr__(
                self, "medium_index", _medium_index(self.medium_index)
            )
        for name in ("x_center_m", "y_center_m", "z_m"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))


@dataclass(frozen=True)
class ArrayInspection:
    key: str
    shape: tuple[int, ...]
    dtype: str
    is_complex: bool
    is_object: bool
    is_finite: bool | None
    value_min: float | None = None
    value_max: float | None = None
    magnitude_min: float | None = None
    magnitude_max: float | None = None

    @property
    def minimum(self) -> float | None:
        """Real minimum, or magnitude minimum for a complex array."""

        return self.magnitude_min if self.is_complex else self.value_min

    @property
    def maximum(self) -> float | None:
        """Real maximum, or magnitude maximum for a complex array."""

        return self.magnitude_max if self.is_complex else self.value_max


@dataclass(frozen=True)
class DatasetInspection:
    path: Path
    format: FieldFormat
    keys: tuple[str, ...]
    arrays: tuple[ArrayInspection, ...]
    scalar_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata_candidates: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema_name: str | None = None
    schema_version: int | None = None
    is_canonical: bool = False
    mapping_confidence: MappingConfidence = MappingConfidence.NONE
    suggested_mapping: ImportMapping | None = None
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "format", _enum(self.format, FieldFormat))
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "arrays", tuple(self.arrays))
        object.__setattr__(
            self, "scalar_metadata", MappingProxyType(dict(self.scalar_metadata))
        )
        object.__setattr__(
            self,
            "metadata_candidates",
            MappingProxyType(
                {
                    key: tuple(values)
                    for key, values in self.metadata_candidates.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "mapping_confidence",
            _enum(self.mapping_confidence, MappingConfidence),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def array(self, key: str) -> ArrayInspection:
        for item in self.arrays:
            if item.key == key:
                return item
        raise KeyError(f"No inspected array named {key!r}.")


@dataclass(frozen=True)
class ImportResult:
    field: Field2D
    inspection: DatasetInspection
    mapping: ImportMapping | None = None
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.field, Field2D):
            raise TypeError("field must be a Field2D.")
        if not isinstance(self.inspection, DatasetInspection):
            raise TypeError("inspection must be a DatasetInspection.")
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True)
class ExportOptions:
    compressed: bool = True
    atomic: bool = True
    include_metadata: bool = True

    def __post_init__(self) -> None:
        for name in ("compressed", "atomic", "include_metadata"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")


@dataclass(frozen=True)
class ExportReport:
    path: Path
    format: FieldFormat
    schema_name: str
    schema_version: int
    bytes_written: int
    sha256: str
    atomic: bool
    compressed: bool
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "format", _enum(self.format, FieldFormat))
        if self.bytes_written < 0:
            raise ValueError("bytes_written must be non-negative.")
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _key_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    try:
        normalized = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of keys.") from exc
    if not all(isinstance(key, str) and key.strip() for key in normalized):
        raise ValueError(f"{name} must contain non-empty string keys.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate keys.")
    return tuple(key.strip() for key in normalized)


def _optional_key(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _enum(value: Any, enum_type: type[Enum]):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {enum_type.__name__}: {value!r}.") from exc


def _optional_enum(value: Any, enum_type: type[Enum]):
    return None if value is None else _enum(value, enum_type)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_float(value: object, name: str) -> float:
    normalized = _finite_float(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _medium_index(value: object) -> complex:
    try:
        normalized = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("medium_index must be complex-compatible.") from exc
    if (
        not math.isfinite(normalized.real)
        or not math.isfinite(normalized.imag)
        or normalized.real <= 0
    ):
        raise ValueError("medium_index must be finite with positive real part.")
    return normalized


__all__ = [
    "AmplitudeConvention",
    "ArrayInspection",
    "AxisOrder",
    "DatasetInspection",
    "ExportOptions",
    "ExportReport",
    "FieldFormat",
    "ImportMapping",
    "ImportResult",
    "LengthUnit",
    "MappingConfidence",
    "PhaseUnit",
]
