"""Composable, zero-thickness optical-element layers.

The package uses the ``exp(-i omega t)`` convention.  Consequently a positive
thin lens that focuses in +Z has the paraxial transmission

``exp(-i k r**2 / (2 f))``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional

from ..core import (
    Field2D,
    Grid2D,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from .layers import LayerBase, RunContext, _deep_freeze, _json_value


class ApertureShape(str, Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"


class MaskSamplingMode(str, Enum):
    """How a sampled complex mask is aligned to the incident field."""

    STRICT = "strict"
    RESAMPLE_COMPLEX = "resample_complex"


class LensPhaseModel(str, Enum):
    """Phase profile used by :class:`IdealLensLayer`."""

    EXACT_EQUAL_PATH = "exact_equal_path"
    PARAXIAL = "paraxial"


@dataclass(frozen=True)
class ApertureSpec:
    """Analytic aperture evaluated at field sample centres.

    ``size_x_m`` and ``size_y_m`` are full diameters/widths, not radii.
    Rotation is counter-clockwise in the XY plane.
    """

    shape: ApertureShape
    size_x_m: float
    size_y_m: float | None = None
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    rotation_rad: float = 0.0
    invert: bool = False
    edge_inclusive: bool = True

    def __post_init__(self) -> None:
        try:
            shape = (
                self.shape
                if isinstance(self.shape, ApertureShape)
                else ApertureShape(self.shape)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "shape must be 'circle', 'rectangle', or 'ellipse'."
            ) from exc
        size_x = _positive_float(self.size_x_m, "size_x_m")
        size_y = (
            size_x
            if self.size_y_m is None
            else _positive_float(self.size_y_m, "size_y_m")
        )
        if shape is ApertureShape.CIRCLE and not math.isclose(
            size_x, size_y, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("A circular aperture requires equal X/Y diameters.")
        if not isinstance(self.invert, bool):
            raise TypeError("invert must be a bool.")
        if not isinstance(self.edge_inclusive, bool):
            raise TypeError("edge_inclusive must be a bool.")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "size_x_m", size_x)
        object.__setattr__(self, "size_y_m", size_y)
        object.__setattr__(
            self, "center_x_m", _finite_float(self.center_x_m, "center_x_m")
        )
        object.__setattr__(
            self, "center_y_m", _finite_float(self.center_y_m, "center_y_m")
        )
        object.__setattr__(
            self, "rotation_rad", _finite_float(self.rotation_rad, "rotation_rad")
        )

    def mask(
        self,
        grid: Grid2D,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a real aperture mask with shape ``[ny, nx]``."""

        if not isinstance(grid, Grid2D):
            raise TypeError("grid must be a Grid2D.")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64.")
        x, y = grid.meshgrid(device=device, dtype=dtype)
        dx = x - self.center_x_m
        dy = y - self.center_y_m
        cosine = math.cos(self.rotation_rad)
        sine = math.sin(self.rotation_rad)
        x_local = cosine * dx + sine * dy
        y_local = -sine * dx + cosine * dy

        if self.shape is ApertureShape.RECTANGLE:
            x_value = torch.abs(x_local)
            y_value = torch.abs(y_local)
            if self.edge_inclusive:
                inside = (x_value <= self.size_x_m / 2) & (
                    y_value <= self.size_y_m / 2
                )
            else:
                inside = (x_value < self.size_x_m / 2) & (
                    y_value < self.size_y_m / 2
                )
        else:
            normalized = (x_local / (self.size_x_m / 2)) ** 2 + (
                y_local / (self.size_y_m / 2)
            ) ** 2
            inside = normalized <= 1 if self.edge_inclusive else normalized < 1
        if self.invert:
            inside = ~inside
        return inside.to(dtype=dtype)

    def to_config(self) -> dict[str, Any]:
        return {
            "shape": self.shape.value,
            "size_x_m": self.size_x_m,
            "size_y_m": self.size_y_m,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "rotation_rad": self.rotation_rad,
            "invert": self.invert,
            "edge_inclusive": self.edge_inclusive,
        }


@dataclass(frozen=True, kw_only=True)
class ThinElementLayer(LayerBase):
    """Base for multiplicative zero-thickness optical elements."""

    def evaluate_transmission(self, field: Field2D) -> torch.Tensor:
        raise NotImplementedError

    def apply(
        self,
        field: Field2D,
        context: RunContext | None = None,
    ) -> Field2D:
        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        if not self.enabled:
            return field
        if context is not None and not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext or None.")
        if context is not None and context.cancel_token is not None:
            context.cancel_token.throw_if_cancelled()
        report = self.validate(field)
        report.raise_for_errors()
        transmission = self.evaluate_transmission(field)
        if not isinstance(transmission, torch.Tensor):
            raise TypeError("transmission() must return a torch.Tensor.")
        if transmission.ndim == 2:
            transmission = transmission.unsqueeze(0)
        if transmission.ndim != 3:
            raise ValueError("Transmission must have shape [ny,nx] or [C,ny,nx].")
        if transmission.shape[-2:] != field.grid.shape:
            raise ValueError("Transmission spatial shape must match the field grid.")
        if transmission.shape[0] not in (1, field.component_count):
            raise ValueError(
                "Transmission component count must be one or match the field."
            )
        output = field.data * transmission.to(
            device=field.device,
            dtype=field.dtype,
        )
        if context is not None and context.cancel_token is not None:
            context.cancel_token.throw_if_cancelled()
        return field.with_data(output)


@dataclass(frozen=True)
class ApertureLayer(ThinElementLayer):
    aperture: ApertureSpec
    name: str = field(default="Aperture", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.aperture, ApertureSpec):
            raise TypeError("aperture must be an ApertureSpec.")

    def parameter_config(self) -> dict[str, Any]:
        return {"version": 1, "aperture": self.aperture.to_config()}

    def evaluate_transmission(self, field: Field2D) -> torch.Tensor:
        return self.aperture.mask(
            field.grid,
            device=field.device,
            dtype=field.real_dtype,
        )


@dataclass(frozen=True)
class IdealLensLayer(ThinElementLayer):
    focal_length_m: float
    design_wavelength_m: float | None = None
    phase_model: LensPhaseModel = LensPhaseModel.EXACT_EQUAL_PATH
    transmission_amplitude: float = 1.0
    phase_offset_rad: float = 0.0
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    medium_index: complex | None = None
    aperture: ApertureSpec | None = None
    name: str = field(default="Ideal lens", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        focal_length = _finite_float(self.focal_length_m, "focal_length_m")
        if focal_length == 0.0:
            raise ValueError("focal_length_m must be non-zero.")
        design_wavelength = (
            None
            if self.design_wavelength_m is None
            else _positive_float(
                self.design_wavelength_m, "design_wavelength_m"
            )
        )
        transmission_amplitude = _finite_float(
            self.transmission_amplitude, "transmission_amplitude"
        )
        if not 0.0 <= transmission_amplitude <= 1.0:
            raise ValueError("transmission_amplitude must lie in [0, 1].")
        try:
            phase_model = (
                self.phase_model
                if isinstance(self.phase_model, LensPhaseModel)
                else LensPhaseModel(self.phase_model)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "phase_model must be 'exact_equal_path' or 'paraxial'."
            ) from exc
        medium_index = (
            None
            if self.medium_index is None
            else _medium_index(self.medium_index)
        )
        if self.aperture is not None and not isinstance(
            self.aperture, ApertureSpec
        ):
            raise TypeError("aperture must be an ApertureSpec or None.")
        object.__setattr__(self, "focal_length_m", focal_length)
        object.__setattr__(self, "design_wavelength_m", design_wavelength)
        object.__setattr__(
            self, "transmission_amplitude", transmission_amplitude
        )
        object.__setattr__(self, "phase_model", phase_model)
        object.__setattr__(
            self,
            "phase_offset_rad",
            _finite_float(self.phase_offset_rad, "phase_offset_rad"),
        )
        object.__setattr__(
            self, "center_x_m", _finite_float(self.center_x_m, "center_x_m")
        )
        object.__setattr__(
            self, "center_y_m", _finite_float(self.center_y_m, "center_y_m")
        )
        object.__setattr__(self, "medium_index", medium_index)

    def parameter_config(self) -> dict[str, Any]:
        return {
            "version": 1,
            "focal_length_m": self.focal_length_m,
            "phase_model": self.phase_model.value,
            "design_wavelength_m": self.design_wavelength_m,
            "transmission_amplitude": self.transmission_amplitude,
            "phase_offset_rad": self.phase_offset_rad,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "medium_index": _json_value(self.medium_index),
            "aperture": (
                None if self.aperture is None else self.aperture.to_config()
            ),
        }

    def validate(self, field: Field2D) -> ValidationReport:
        report = super().validate(field)
        issues = list(report.issues)
        if (
            self.design_wavelength_m is not None
            and not math.isclose(
                field.wavelength_m,
                self.design_wavelength_m,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            issues.append(
                _issue(
                    self,
                    Severity.INFO,
                    "lens.off_design_wavelength",
                    "The field wavelength differs from the lens design wavelength.",
                    "design_wavelength_m",
                    {
                        "field_wavelength_m": field.wavelength_m,
                        "design_wavelength_m": self.design_wavelength_m,
                    },
                )
            )
        return ValidationReport(tuple(issues))

    def evaluate_transmission(self, field: Field2D) -> torch.Tensor:
        x, y = field.grid.meshgrid(
            device=field.device,
            dtype=field.real_dtype,
        )
        radius_squared = (x - self.center_x_m) ** 2 + (
            y - self.center_y_m
        ) ** 2
        wavelength = self.design_wavelength_m or field.wavelength_m
        medium = self.medium_index or field.medium_index
        # A thin phase element uses the phase index.  Absorption belongs in an
        # explicit amplitude mask or propagation layer.
        wave_number = 2 * math.pi * medium.real / wavelength
        if self.phase_model is LensPhaseModel.PARAXIAL:
            lens_phase = -wave_number * radius_squared / (
                2 * self.focal_length_m
            )
        else:
            optical_path_difference = (
                torch.sqrt(self.focal_length_m**2 + radius_squared)
                - abs(self.focal_length_m)
            )
            lens_phase = (
                -math.copysign(1.0, self.focal_length_m)
                * wave_number
                * optical_path_difference
            )
        phase = self.phase_offset_rad + lens_phase
        transmission = self.transmission_amplitude * torch.exp(1j * phase)
        if self.aperture is not None:
            transmission = transmission * self.aperture.mask(
                field.grid,
                device=field.device,
                dtype=field.real_dtype,
            )
        return transmission


@dataclass(frozen=True)
class Binary2LensLayer(ThinElementLayer):
    """Radial even-power phase, matching the legacy ``Binary2Phase`` convention.

    ``phi = phase_offset_rad + sum(C[i-1] * r_mm**(2*i), i=1..N)``,
    where ``r_mm`` is the numerical radius in millimetres relative to
    ``center_x_m, center_y_m``. Coefficients have units rad/mm**(2*i).
    They are NOT waves, normalized-radius coefficients, or two-level phases.

    The coefficients directly specify a fixed phase mask: wavelength and
    refractive index do not implicitly rescale it. A physical dispersion model
    or a meta-atom database must be specified separately for that purpose.
    One scalar transmission is applied to every input component. Coefficients
    are immutable serialized constants; input-field autograd is preserved.
    """

    coefficients: tuple[float, ...]
    transmission_amplitude: float = 1.0
    phase_offset_rad: float = 0.0
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    aperture: ApertureSpec | None = None
    name: str = field(default="Binary2 lens", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.coefficients, (list, tuple)):
            raise TypeError("coefficients must be a non-empty list or tuple of real numbers.")
        if not self.coefficients:
            raise ValueError("coefficients must contain at least C1 (the r^2 coefficient).")
        coefficients = tuple(
            _finite_float(value, f"coefficients[{i}]")
            for i, value in enumerate(self.coefficients)
        )
        amplitude = _finite_float(self.transmission_amplitude, "transmission_amplitude")
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError("transmission_amplitude must lie in [0, 1].")
        if self.aperture is not None and not isinstance(self.aperture, ApertureSpec):
            raise TypeError("aperture must be an ApertureSpec or None.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "transmission_amplitude", amplitude)
        for key in ("phase_offset_rad", "center_x_m", "center_y_m"):
            object.__setattr__(self, key, _finite_float(getattr(self, key), key))

    def parameter_config(self) -> dict[str, Any]:
        return {
            "version": 1,
            "coefficients": list(self.coefficients),
            "transmission_amplitude": self.transmission_amplitude,
            "phase_offset_rad": self.phase_offset_rad,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "aperture": None if self.aperture is None else self.aperture.to_config(),
        }

    def evaluate_transmission(self, field: Field2D) -> torch.Tensor:
        x, y = field.grid.meshgrid(device=field.device, dtype=field.real_dtype)
        radius_squared_mm = ((x - self.center_x_m) * 1e3) ** 2 + (
            (y - self.center_y_m) * 1e3
        ) ** 2
        coefficients = torch.as_tensor(
            self.coefficients, device=field.device, dtype=field.real_dtype
        )
        # Horner form uses one spatial map, not an [ny,nx,coefficient] volume.
        phase = torch.zeros_like(radius_squared_mm)
        for coefficient in reversed(coefficients.unbind()):
            phase = (phase + coefficient) * radius_squared_mm
        phase = phase + self.phase_offset_rad
        if not torch.isfinite(phase).all().item():
            raise ValueError("Binary2 phase overflow: check coefficients, mm units, grid and dtype.")
        transmission = self.transmission_amplitude * torch.exp(1j * phase)
        if self.aperture is not None:
            transmission = transmission * self.aperture.mask(
                field.grid, device=field.device, dtype=field.real_dtype
            )
        return transmission


@dataclass(frozen=True)
class ComplexMaskLayer(ThinElementLayer):
    transmission: torch.Tensor
    grid: Grid2D
    sampling_mode: MaskSamplingMode = MaskSamplingMode.STRICT
    asset_reference: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    name: str = field(default="Complex mask", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.transmission, torch.Tensor):
            raise TypeError("transmission must be a torch.Tensor.")
        if self.transmission.dtype not in (
            torch.complex64,
            torch.complex128,
        ):
            raise TypeError(
                "transmission must use torch.complex64 or torch.complex128."
            )
        # Snapshot caller-owned storage while preserving autograd connectivity.
        data = self.transmission.clone()
        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim != 3:
            raise ValueError(
                "transmission must have shape [ny,nx] or [C,ny,nx]."
            )
        if not isinstance(self.grid, Grid2D):
            raise TypeError("grid must be a Grid2D.")
        if data.shape[-2:] != self.grid.shape:
            raise ValueError("transmission spatial shape must match grid.")
        if data.shape[0] not in (1, 2):
            raise ValueError("transmission must have one or two components.")
        try:
            sampling_mode = (
                self.sampling_mode
                if isinstance(self.sampling_mode, MaskSamplingMode)
                else MaskSamplingMode(self.sampling_mode)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sampling_mode must be 'strict' or 'resample_complex'."
            ) from exc
        asset_reference = self.asset_reference
        if asset_reference is not None:
            if not isinstance(asset_reference, str):
                raise TypeError("asset_reference must be a string or None.")
            asset_reference = asset_reference.strip()
            if not asset_reference:
                raise ValueError("asset_reference must not be empty.")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")
        frozen_provenance = _deep_freeze(self.provenance)
        _json_value(frozen_provenance)
        object.__setattr__(self, "transmission", data)
        object.__setattr__(self, "sampling_mode", sampling_mode)
        object.__setattr__(self, "asset_reference", asset_reference)
        object.__setattr__(self, "provenance", frozen_provenance)

    @classmethod
    def from_amplitude_phase(
        cls,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        grid: Grid2D,
        **kwargs: Any,
    ) -> "ComplexMaskLayer":
        """Build a complex mask from real amplitude and phase tensors."""

        transmission = _complex_from_amplitude_phase(amplitude, phase)
        return cls(transmission=transmission, grid=grid, **kwargs)

    @classmethod
    def from_amplitude(
        cls,
        amplitude: torch.Tensor,
        grid: Grid2D,
        **kwargs: Any,
    ) -> "ComplexMaskLayer":
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError("amplitude must be a torch.Tensor.")
        return cls.from_amplitude_phase(
            amplitude,
            torch.zeros_like(amplitude),
            grid,
            **kwargs,
        )

    @classmethod
    def from_phase(
        cls,
        phase: torch.Tensor,
        grid: Grid2D,
        **kwargs: Any,
    ) -> "ComplexMaskLayer":
        if not isinstance(phase, torch.Tensor):
            raise TypeError("phase must be a torch.Tensor.")
        return cls.from_amplitude_phase(
            torch.ones_like(phase),
            phase,
            grid,
            **kwargs,
        )

    def parameter_config(self) -> dict[str, Any]:
        return {
            "version": 1,
            "grid": _json_value(self.grid),
            "sampling_mode": self.sampling_mode.value,
            "asset_reference": self.asset_reference,
            "provenance": _json_value(self.provenance),
            "transmission": {
                "shape": list(self.transmission.shape),
                "dtype": str(self.transmission.dtype),
                "sha256": _tensor_sha256(self.transmission),
            },
        }

    def validate(self, field: Field2D) -> ValidationReport:
        report = super().validate(field)
        issues = list(report.issues)
        detached = self.transmission.detach()
        finite = torch.isfinite(detached.real) & torch.isfinite(detached.imag)
        if not torch.all(finite).item():
            issues.append(
                _issue(
                    self,
                    Severity.ERROR,
                    "mask.nonfinite_transmission",
                    "Complex mask transmission contains NaN or infinity.",
                    "transmission",
                )
            )
        elif torch.any(torch.abs(detached) > 1.0).item():
            issues.append(
                _issue(
                    self,
                    Severity.WARNING,
                    "mask.gain_transmission",
                    (
                        "Complex mask magnitude exceeds one; this represents "
                        "an active/gain mask rather than a passive element."
                    ),
                    "transmission",
                    {"maximum_magnitude": float(torch.abs(detached).max().item())},
                )
            )
        if self.transmission.shape[0] not in (
            1,
            field.component_count,
        ):
            issues.append(
                _issue(
                    self,
                    Severity.ERROR,
                    "mask.component_mismatch",
                    "Mask components must be one or match the field components.",
                    "transmission",
                    {
                        "mask_components": self.transmission.shape[0],
                        "field_components": field.component_count,
                    },
                )
            )
        if self.sampling_mode is MaskSamplingMode.STRICT and self.grid != field.grid:
            issues.append(
                _issue(
                    self,
                    Severity.ERROR,
                    "mask.grid_mismatch",
                    "Strict mask sampling requires an identical field grid.",
                    "grid",
                    {
                        "mask_shape": self.grid.shape,
                        "field_shape": field.grid.shape,
                    },
                )
            )
        elif (
            self.sampling_mode is MaskSamplingMode.RESAMPLE_COMPLEX
            and not _extent_contains(self.grid.pixel_extent, field.grid.pixel_extent)
        ):
            issues.append(
                _issue(
                    self,
                    Severity.WARNING,
                    "mask.resample_outside_extent",
                    "Part of the field grid lies outside the mask and will be zero.",
                    "grid",
                )
            )
        return ValidationReport(tuple(issues))

    def evaluate_transmission(self, field: Field2D) -> torch.Tensor:
        data = self.transmission.to(
            device=field.device,
            dtype=field.dtype,
        )
        if self.grid == field.grid:
            return data
        if self.sampling_mode is MaskSamplingMode.STRICT:
            # Direct callers get the same structured error as system runs.
            self.validate(field).raise_for_errors()
        return _resample_complex(data, self.grid, field.grid)


def _resample_complex(
    data: torch.Tensor,
    input_grid: Grid2D,
    output_grid: Grid2D,
) -> torch.Tensor:
    real_dtype = torch.float32 if data.dtype == torch.complex64 else torch.float64
    x, y = output_grid.meshgrid(device=data.device, dtype=real_dtype)
    x_index = (
        (x - input_grid.x_center) / input_grid.dx
        + (input_grid.nx - 1) / 2
    )
    y_index = (
        (y - input_grid.y_center) / input_grid.dy
        + (input_grid.ny - 1) / 2
    )
    x_normalized = (
        torch.zeros_like(x_index)
        if input_grid.nx == 1
        else 2 * x_index / (input_grid.nx - 1) - 1
    )
    y_normalized = (
        torch.zeros_like(y_index)
        if input_grid.ny == 1
        else 2 * y_index / (input_grid.ny - 1) - 1
    )
    sample_grid = torch.stack((x_normalized, y_normalized), dim=-1).unsqueeze(0)
    real = torch_functional.grid_sample(
        data.real.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    imag = torch_functional.grid_sample(
        data.imag.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    return torch.complex(real, imag)


def _complex_from_amplitude_phase(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(amplitude, torch.Tensor):
        raise TypeError("amplitude must be a torch.Tensor.")
    if not isinstance(phase, torch.Tensor):
        raise TypeError("phase must be a torch.Tensor.")
    if amplitude.shape != phase.shape:
        raise ValueError("amplitude and phase must have identical shapes.")
    if amplitude.is_complex() or phase.is_complex():
        raise TypeError("amplitude and phase must be real tensors.")
    if amplitude.device != phase.device:
        phase = phase.to(device=amplitude.device)
    if torch.any(amplitude < 0).item():
        raise ValueError("amplitude must be non-negative.")
    real_dtype = (
        torch.float64
        if amplitude.dtype == torch.float64 or phase.dtype == torch.float64
        else torch.float32
    )
    amplitude = amplitude.to(dtype=real_dtype)
    phase = phase.to(device=amplitude.device, dtype=real_dtype)
    return amplitude * torch.exp(1j * phase)


def _issue(
    layer: LayerBase,
    severity: Severity,
    code: str,
    message: str,
    parameter_path: str,
    details: Mapping[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        layer_id=str(layer.id),
        parameter_path=parameter_path,
        details={} if details is None else details,
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().resolve_conj().resolve_neg().contiguous().cpu()
    payload = data.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _extent_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] >= inner[1]
        and outer[2] <= inner[2]
        and outer[3] >= inner[3]
    )


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
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _medium_index(value: object) -> complex:
    if isinstance(value, bool):
        raise TypeError("medium_index must be a real or complex number.")
    try:
        normalized = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("medium_index must be a real or complex number.") from exc
    if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
        raise ValueError("medium_index must be finite.")
    if normalized.real <= 0.0:
        raise ValueError("medium_index.real must be positive.")
    return normalized


__all__ = [
    "ApertureLayer",
    "ApertureShape",
    "ApertureSpec",
    "Binary2LensLayer",
    "ComplexMaskLayer",
    "IdealLensLayer",
    "LensPhaseModel",
    "MaskSamplingMode",
    "ThinElementLayer",
]
