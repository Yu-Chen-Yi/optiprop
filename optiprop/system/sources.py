"""Serializable coherent-field source factories.

``IncidentSource`` intentionally is not an ``OpticalLayer``: it creates the
field that enters an :class:`~optiprop.system.OpticalSystem`, so it does not
require a dummy input field and cannot accidentally replace a field halfway
through a system run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import torch

from ..core import (
    EX_EY_COMPONENTS,
    SCALAR_COMPONENTS,
    Field2D,
    Grid2D,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from .layers import FieldGeometry, _deep_freeze, _json_value
from .thin_elements import ApertureSpec, _medium_index, _tensor_sha256


class SourceKind(str, Enum):
    PLANE_WAVE = "plane_wave"
    TILTED_PLANE_WAVE = "tilted_plane_wave"
    GAUSSIAN = "gaussian"
    ELLIPTICAL_GAUSSIAN = "elliptical_gaussian"
    IMPORTED_FIELD = "imported_field"


@dataclass(frozen=True)
class IncidentSource:
    """Frozen analytic/imported source definition for the Source panel."""

    kind: SourceKind
    grid: Grid2D | None = None
    wavelength_m: float | None = None
    medium_index: complex = 1.0
    components: tuple[str, ...] = SCALAR_COMPONENTS
    amplitudes: tuple[complex, ...] = (1.0 + 0.0j,)
    phase_offset_rad: float = 0.0
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    waist_x_m: float | None = None
    waist_y_m: float | None = None
    angle_x_rad: float = 0.0
    angle_y_rad: float = 0.0
    aperture: ApertureSpec | None = None
    source_field: Field2D | None = field(default=None, repr=False, compare=False)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4, kw_only=True)
    name: str = field(default="Source", kw_only=True)
    enabled: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        try:
            kind = (
                self.kind
                if isinstance(self.kind, SourceKind)
                else SourceKind(self.kind)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown source kind {self.kind!r}.") from exc
        source_id = _uuid(self.id)
        name = _required_text(self.name, "name")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        if not isinstance(self.components, (tuple, list)):
            raise TypeError("components must be a component-name sequence.")
        components = tuple(self.components)
        if components not in (SCALAR_COMPONENTS, EX_EY_COMPONENTS):
            raise ValueError(
                "components must be ('scalar',) or ('Ex', 'Ey')."
            )
        amplitudes = _component_amplitudes(self.amplitudes, len(components))
        medium_index = _medium_index(self.medium_index)
        aperture = self.aperture
        if aperture is not None and not isinstance(aperture, ApertureSpec):
            raise TypeError("aperture must be an ApertureSpec or None.")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")
        provenance = _deep_freeze(self.provenance)
        _json_value(provenance)

        source_field = self.source_field
        if kind is SourceKind.IMPORTED_FIELD:
            if not isinstance(source_field, Field2D):
                raise TypeError(
                    "source_field must be a Field2D for imported_field sources."
                )
            # Snapshot the caller's tensor while retaining a CloneBackward
            # edge when it participates in an inverse-design graph.
            source_field = source_field.with_data(source_field.data.clone())
            grid = source_field.grid
            wavelength = source_field.wavelength_m
            medium_index = source_field.medium_index
            components = source_field.components
            amplitudes = tuple(1.0 + 0.0j for _ in components)
        else:
            if self.source_field is not None:
                raise ValueError(
                    "source_field is accepted only for imported_field sources."
                )
            if not isinstance(self.grid, Grid2D):
                raise TypeError("grid must be a Grid2D for analytic sources.")
            grid = self.grid
            wavelength = _positive_float(self.wavelength_m, "wavelength_m")

        waist_x = (
            None
            if self.waist_x_m is None
            else _positive_float(self.waist_x_m, "waist_x_m")
        )
        waist_y = (
            None
            if self.waist_y_m is None
            else _positive_float(self.waist_y_m, "waist_y_m")
        )
        if kind in (SourceKind.GAUSSIAN, SourceKind.ELLIPTICAL_GAUSSIAN):
            if waist_x is None:
                raise ValueError("Gaussian sources require waist_x_m.")
            if waist_y is None:
                waist_y = waist_x
        phase_offset = _finite_float(self.phase_offset_rad, "phase_offset_rad")
        center_x = _finite_float(self.center_x_m, "center_x_m")
        center_y = _finite_float(self.center_y_m, "center_y_m")
        angle_x = _finite_float(self.angle_x_rad, "angle_x_rad")
        angle_y = _finite_float(self.angle_y_rad, "angle_y_rad")
        if angle_x != 0.0 or angle_y != 0.0:
            transverse_fraction = math.sin(angle_x) ** 2 + math.sin(angle_y) ** 2
            if transverse_fraction > 1.0 + 1e-12:
                raise ValueError(
                    "Tilt angles imply a transverse wave vector larger than k."
                )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", source_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "wavelength_m", wavelength)
        object.__setattr__(self, "medium_index", medium_index)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "phase_offset_rad", phase_offset)
        object.__setattr__(self, "center_x_m", center_x)
        object.__setattr__(self, "center_y_m", center_y)
        object.__setattr__(self, "waist_x_m", waist_x)
        object.__setattr__(self, "waist_y_m", waist_y)
        object.__setattr__(self, "angle_x_rad", angle_x)
        object.__setattr__(self, "angle_y_rad", angle_y)
        object.__setattr__(self, "source_field", source_field)
        object.__setattr__(self, "provenance", provenance)
        json.dumps(self.parameter_config(), sort_keys=True, allow_nan=False)

    @classmethod
    def plane_wave(
        cls,
        grid: Grid2D,
        wavelength_m: float,
        **kwargs: Any,
    ) -> "IncidentSource":
        return cls(
            kind=SourceKind.PLANE_WAVE,
            grid=grid,
            wavelength_m=wavelength_m,
            **kwargs,
        )

    @classmethod
    def tilted_plane_wave(
        cls,
        grid: Grid2D,
        wavelength_m: float,
        angle_x_rad: float,
        angle_y_rad: float,
        **kwargs: Any,
    ) -> "IncidentSource":
        return cls(
            kind=SourceKind.TILTED_PLANE_WAVE,
            grid=grid,
            wavelength_m=wavelength_m,
            angle_x_rad=angle_x_rad,
            angle_y_rad=angle_y_rad,
            **kwargs,
        )

    @classmethod
    def gaussian(
        cls,
        grid: Grid2D,
        wavelength_m: float,
        waist_x_m: float,
        waist_y_m: float | None = None,
        **kwargs: Any,
    ) -> "IncidentSource":
        kind = (
            SourceKind.GAUSSIAN
            if waist_y_m is None or waist_y_m == waist_x_m
            else SourceKind.ELLIPTICAL_GAUSSIAN
        )
        return cls(
            kind=kind,
            grid=grid,
            wavelength_m=wavelength_m,
            waist_x_m=waist_x_m,
            waist_y_m=waist_y_m,
            **kwargs,
        )

    @classmethod
    def from_field(
        cls,
        field: Field2D,
        *,
        provenance: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "IncidentSource":
        return cls(
            kind=SourceKind.IMPORTED_FIELD,
            source_field=field,
            provenance={} if provenance is None else provenance,
            **kwargs,
        )

    @classmethod
    def from_amplitude_phase(
        cls,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        grid: Grid2D,
        wavelength_m: float,
        *,
        medium_index: complex = 1.0,
        components: Sequence[str] | None = None,
        provenance: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "IncidentSource":
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError("amplitude must be a torch.Tensor.")
        if not isinstance(phase, torch.Tensor):
            raise TypeError("phase must be a torch.Tensor.")
        if amplitude.shape != phase.shape:
            raise ValueError("amplitude and phase must have identical shapes.")
        if amplitude.ndim == 2:
            amplitude = amplitude.unsqueeze(0)
            phase = phase.unsqueeze(0)
        if amplitude.ndim != 3 or amplitude.shape[-2:] != grid.shape:
            raise ValueError(
                "amplitude/phase must have shape [ny,nx] or [C,ny,nx]."
            )
        if amplitude.shape[0] not in (1, 2):
            raise ValueError("amplitude/phase must have one or two components.")
        if amplitude.is_complex() or phase.is_complex():
            raise TypeError("amplitude and phase must be real tensors.")
        if not torch.all(torch.isfinite(amplitude)).item():
            raise ValueError("amplitude must contain only finite values.")
        if not torch.all(torch.isfinite(phase)).item():
            raise ValueError("phase must contain only finite values.")
        if torch.any(amplitude < 0).item():
            raise ValueError("amplitude must be non-negative.")
        real_dtype = (
            torch.float64
            if amplitude.dtype == torch.float64 or phase.dtype == torch.float64
            else torch.float32
        )
        amplitude = amplitude.to(real_dtype)
        phase = phase.to(device=amplitude.device, dtype=real_dtype)
        data = amplitude * torch.exp(1j * phase)
        resolved_components = (
            tuple(components)
            if components is not None
            else (
                SCALAR_COMPONENTS
                if data.shape[0] == 1
                else EX_EY_COMPONENTS
            )
        )
        field_value = Field2D(
            data=data,
            grid=grid,
            wavelength_m=wavelength_m,
            medium_index=medium_index,
            components=resolved_components,
        )
        return cls.from_field(
            field_value,
            provenance={} if provenance is None else provenance,
            **kwargs,
        )

    @property
    def parameter_hash(self) -> str:
        encoded = json.dumps(
            self.parameter_config(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def parameter_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "version": 1,
            "kind": self.kind.value,
            "grid": _json_value(self.grid),
            "wavelength_m": self.wavelength_m,
            "medium_index": _json_value(self.medium_index),
            "components": list(self.components),
            "amplitudes": [_json_value(value) for value in self.amplitudes],
            "phase_offset_rad": self.phase_offset_rad,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "waist_x_m": self.waist_x_m,
            "waist_y_m": self.waist_y_m,
            "angle_x_rad": self.angle_x_rad,
            "angle_y_rad": self.angle_y_rad,
            "aperture": (
                None if self.aperture is None else self.aperture.to_config()
            ),
            "provenance": _json_value(self.provenance),
        }
        if self.source_field is not None:
            config["imported_field"] = {
                "shape": list(self.source_field.shape),
                "dtype": str(self.source_field.dtype),
                "z_m": self.source_field.z_m,
                "sha256": _tensor_sha256(self.source_field.data),
            }
        return config

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "id": str(self.id),
            "name": self.name,
            "enabled": self.enabled,
            **self.parameter_config(),
        }

    def validate(self) -> ValidationReport:
        issues = []
        if not self.enabled:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="source.disabled",
                    message="A disabled source cannot create an incident field.",
                    layer_id=str(self.id),
                    parameter_path="enabled",
                    suggested_fix="Enable the source before creating a field.",
                )
            )
        if self.medium_index.imag < 0.0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="source.active_medium",
                    message=(
                        "Incident sources in active media are not supported."
                    ),
                    layer_id=str(self.id),
                    parameter_path="medium_index",
                    suggested_fix=(
                        "Use a passive medium with medium_index.imag >= 0."
                    ),
                    details={"medium_index": str(self.medium_index)},
                )
            )
        return ValidationReport(tuple(issues))

    def preview_geometry(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FieldGeometry:
        if self.source_field is not None:
            resolved_device = (
                self.source_field.device
                if device is None
                else torch.device(device)
            )
            resolved_dtype = self.source_field.dtype if dtype is None else dtype
            return FieldGeometry(
                grid=self.source_field.grid,
                wavelength_m=self.source_field.wavelength_m,
                medium_index=self.source_field.medium_index,
                components=self.source_field.components,
                z_m=self.source_field.z_m,
                dtype=_complex_dtype(resolved_dtype),
                device=resolved_device,
            )
        return FieldGeometry(
            grid=self.grid,
            wavelength_m=self.wavelength_m,
            medium_index=self.medium_index,
            components=self.components,
            z_m=0.0,
            dtype=_complex_dtype(dtype or torch.complex64),
            device=torch.device("cpu" if device is None else device),
        )

    def create(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Field2D:
        """Create a new canonical field without requiring a system input."""

        self.validate().raise_for_errors()
        if self.source_field is not None:
            created = self.source_field.to(
                device=device,
                dtype=(
                    self.source_field.dtype
                    if dtype is None
                    else _complex_dtype(dtype)
                ),
                copy=True,
            )
            created = created.with_data(
                created.data,
                metadata={
                    **dict(created.metadata),
                    "source_id": str(self.id),
                    "source_kind": self.kind.value,
                    "source_provenance": self.provenance,
                },
            )
            if self.aperture is None:
                return created
            aperture = self.aperture.mask(
                created.grid,
                device=created.device,
                dtype=created.real_dtype,
            )
            return created.with_data(created.data * aperture[None])
        target_dtype = _complex_dtype(dtype or torch.complex64)
        target_device = torch.device("cpu" if device is None else device)
        real_dtype = (
            torch.float32 if target_dtype == torch.complex64 else torch.float64
        )
        x, y = self.grid.meshgrid(device=target_device, dtype=real_dtype)
        x_offset = x - self.center_x_m
        y_offset = y - self.center_y_m
        phase = torch.full_like(x, self.phase_offset_rad)
        envelope = torch.ones_like(x)
        if self.angle_x_rad != 0.0 or self.angle_y_rad != 0.0:
            wave_number = 2 * math.pi * self.medium_index.real / self.wavelength_m
            phase = phase + wave_number * (
                math.sin(self.angle_x_rad) * x_offset
                + math.sin(self.angle_y_rad) * y_offset
            )
        if self.kind in (
            SourceKind.GAUSSIAN,
            SourceKind.ELLIPTICAL_GAUSSIAN,
        ):
            envelope = torch.exp(
                -((x_offset / self.waist_x_m) ** 2)
                - ((y_offset / self.waist_y_m) ** 2)
            )
        spatial = envelope * torch.exp(1j * phase)
        amplitudes = torch.as_tensor(
            self.amplitudes,
            device=target_device,
            dtype=target_dtype,
        )[:, None, None]
        data = amplitudes * spatial[None]
        if self.aperture is not None:
            data = data * self.aperture.mask(
                self.grid,
                device=target_device,
                dtype=real_dtype,
            )[None]
        return Field2D(
            data=data,
            grid=self.grid,
            wavelength_m=self.wavelength_m,
            medium_index=self.medium_index,
            components=self.components,
            metadata={
                "source_id": str(self.id),
                "source_kind": self.kind.value,
                "source_provenance": self.provenance,
            },
        )

    generate = create


def _component_amplitudes(
    values: object,
    component_count: int,
) -> tuple[complex, ...]:
    if isinstance(values, (int, float, complex)) and not isinstance(values, bool):
        normalized = (complex(values),) * component_count
    else:
        try:
            normalized = tuple(complex(value) for value in values)  # type: ignore[union-attr]
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("amplitudes must be a complex value or sequence.") from exc
        if len(normalized) == 1 and component_count == 2:
            normalized = normalized * 2
    if len(normalized) != component_count:
        raise ValueError("amplitudes must have one value per field component.")
    if not all(
        math.isfinite(value.real) and math.isfinite(value.imag)
        for value in normalized
    ):
        raise ValueError("amplitudes must be finite.")
    return normalized


def _complex_dtype(value: torch.dtype) -> torch.dtype:
    if value not in (torch.complex64, torch.complex128):
        raise TypeError("dtype must be torch.complex64 or torch.complex128.")
    return value


def _uuid(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("id must be a UUID or UUID string.") from exc


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


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
    if value is None:
        raise TypeError(f"{name} must be a real number.")
    normalized = _finite_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return normalized


__all__ = [
    "IncidentSource",
    "SourceKind",
]
