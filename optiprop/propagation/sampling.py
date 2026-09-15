"""Sampling and resource preflight for propagation requests.

This module estimates grids, memory, work, and common numerical risks without
executing a diffraction algorithm.  Findings use stable issue codes so the
same report can drive Python callers, logs, and the future PySide Problems
panel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import torch

from ..core import (
    Field2D,
    Grid2D,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from .base import (
    EvanescentPolicy,
    PaddingMode,
    PrecisionPolicy,
    PropagationMethod,
    PropagationSpec,
)


_FFT_METHODS = frozenset(
    {
        PropagationMethod.ASM,
        PropagationMethod.BLAS,
        PropagationMethod.FRESNEL_TF,
        PropagationMethod.FRESNEL_SCALED,
        PropagationMethod.RS_FFT,
    }
)
_SAME_GRID_METHODS = frozenset(
    {
        PropagationMethod.ASM,
        PropagationMethod.BLAS,
        PropagationMethod.FRESNEL_TF,
        PropagationMethod.RS_FFT,
    }
)


@dataclass(frozen=True)
class SamplingReport:
    """Immutable preflight result for one field/spec pair."""

    method: PropagationMethod
    input_grid: Grid2D
    computational_grid: Grid2D
    output_grid: Grid2D
    wavelength_medium_m: float
    nyquist_x_per_m: float
    nyquist_y_per_m: float
    fresnel_number_x: float | None
    fresnel_number_y: float | None
    edge_intensity_fraction: float | None
    estimated_peak_memory_bytes: int
    estimated_operations: float | None
    recommended_padded_shape: Tuple[int, int]
    validation: ValidationReport = field(default_factory=ValidationReport)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.method, PropagationMethod):
            raise TypeError("method must be a PropagationMethod.")
        for name in ("input_grid", "computational_grid", "output_grid"):
            if not isinstance(getattr(self, name), Grid2D):
                raise TypeError(f"{name} must be a Grid2D.")
        if (
            not math.isfinite(self.wavelength_medium_m)
            or self.wavelength_medium_m <= 0.0
        ):
            raise ValueError("wavelength_medium_m must be positive and finite.")
        if self.estimated_peak_memory_bytes < 0:
            raise ValueError("estimated_peak_memory_bytes must be non-negative.")
        if self.estimated_operations is not None and (
            not math.isfinite(self.estimated_operations)
            or self.estimated_operations < 0.0
        ):
            raise ValueError(
                "estimated_operations must be finite and non-negative."
            )
        if (
            len(self.recommended_padded_shape) != 2
            or any(size <= 0 for size in self.recommended_padded_shape)
        ):
            raise ValueError("recommended_padded_shape must be positive (ny, nx).")
        if not isinstance(self.validation, ValidationReport):
            raise TypeError("validation must be a ValidationReport.")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping.")
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def issues(self) -> Tuple[ValidationIssue, ...]:
        return self.validation.issues

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        return self.validation.errors

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        return self.validation.warnings

    @property
    def report(self) -> ValidationReport:
        """Compatibility alias used by preflight/UI callers."""

        return self.validation

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.input_grid.shape

    @property
    def padded_shape(self) -> Tuple[int, int]:
        return self.computational_grid.shape

    @property
    def physical_extent_m(self) -> Tuple[float, float]:
        return self.input_grid.width, self.input_grid.height

    @property
    def nyquist_frequency_per_m(self) -> Tuple[float, float]:
        return self.nyquist_x_per_m, self.nyquist_y_per_m

    @property
    def medium_wavelength_m(self) -> float:
        return self.wavelength_medium_m

    @property
    def fresnel_number(self) -> Tuple[float | None, float | None]:
        return self.fresnel_number_x, self.fresnel_number_y

    @property
    def estimated_memory_bytes(self) -> int:
        return self.estimated_peak_memory_bytes


def assess_sampling(
    field: Field2D,
    spec: PropagationSpec,
    *,
    layer_id: str | None = None,
    memory_limit_bytes: int | None = None,
    rs_direct_warning_operations: float = 1.0e8,
    rs_direct_error_operations: float = 1.0e10,
    edge_warning_fraction: float = 0.01,
    inspect_edge: bool = True,
) -> SamplingReport:
    """Build a structured numerical preflight without propagating the field."""

    if not isinstance(field, Field2D):
        raise TypeError("field must be a Field2D.")
    if not isinstance(spec, PropagationSpec):
        raise TypeError("spec must be a PropagationSpec.")
    _validate_optional_limit(memory_limit_bytes)
    for value, name in (
        (rs_direct_warning_operations, "rs_direct_warning_operations"),
        (rs_direct_error_operations, "rs_direct_error_operations"),
        (edge_warning_fraction, "edge_warning_fraction"),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if rs_direct_error_operations < rs_direct_warning_operations:
        raise ValueError(
            "rs_direct_error_operations must be at least the warning threshold."
        )
    if edge_warning_fraction > 1.0:
        raise ValueError("edge_warning_fraction must not exceed 1.")

    issues: list[ValidationIssue] = []
    medium_index = spec.resolved_medium_index(field)
    wavelength_medium_m = field.wavelength_m / medium_index.real
    recommended_shape = _recommended_padded_shape(field.grid, spec)
    computational_grid = _computational_grid(
        field.grid, spec, recommended_shape, issues, layer_id
    )
    output_grid = _output_grid(
        field.grid,
        computational_grid,
        wavelength_medium_m,
        spec,
    )

    if spec.distance_m == 0.0:
        issues.append(
            _issue(
                Severity.INFO,
                "sampling.zero_distance",
                "Zero propagation distance is an identity operation.",
                layer_id,
                "distance_m",
                details={"distance_m": 0.0},
            )
        )
    elif spec.distance_m < 0.0:
        issues.append(
            _issue(
                Severity.INFO,
                "sampling.negative_distance",
                "Negative distance requests backward propagation.",
                layer_id,
                "distance_m",
                details={"distance_m": spec.distance_m},
            )
        )
        if medium_index.imag > 0.0:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "sampling.lossy_medium_backward_unstable",
                    "Backward propagation through a lossy medium amplifies "
                    "attenuated components exponentially.",
                    layer_id,
                    "distance_m",
                    suggested_fix=(
                        "Use a forward distance or a real refractive index."
                    ),
                    details={
                        "distance_m": spec.distance_m,
                        "medium_index_imag": medium_index.imag,
                    },
                )
            )
        if (
            spec.method is PropagationMethod.ASM
            and spec.evanescent_policy is EvanescentPolicy.KEEP
        ):
            max_exponent = _maximum_evanescent_exponent(
                field,
                wavelength_medium_m,
                spec.distance_m,
            )
            overflow_limit = (
                80.0
                if _resolved_complex_item_size(field, spec) == 8
                else 700.0
            )
            severity = (
                Severity.ERROR
                if max_exponent > overflow_limit
                else Severity.WARNING
            )
            issues.append(
                _issue(
                    severity,
                    (
                        "sampling.evanescent_backward_overflow"
                        if severity is Severity.ERROR
                        else "sampling.evanescent_backward_ill_conditioned"
                    ),
                    "Exact backward continuation of evanescent components "
                    "is ill-conditioned and may amplify noise.",
                    layer_id,
                    "evanescent_policy",
                    suggested_fix=(
                        "Use the decay or discard policy for stable backpropagation."
                    ),
                    details={
                        "policy": spec.evanescent_policy.value,
                        "maximum_amplification_exponent": max_exponent,
                        "overflow_limit": overflow_limit,
                    },
                )
            )

    if medium_index.imag < 0.0:
        issues.append(
            _issue(
                Severity.ERROR,
                "sampling.active_medium_unsupported",
                "Negative imaginary refractive index represents gain under "
                "the exp(-i omega t) convention and is not supported.",
                layer_id,
                "medium_index",
                suggested_fix="Use a passive medium with Im(n) >= 0.",
                details={"real": medium_index.real, "imag": medium_index.imag},
            )
        )
    elif medium_index.imag != 0.0:
        issues.append(
            _issue(
                Severity.WARNING,
                "sampling.complex_index_approximation",
                "Sampling estimates use the real part of the refractive index.",
                layer_id,
                "medium_index",
                details={"real": medium_index.real, "imag": medium_index.imag},
            )
        )
        if (
            spec.method is PropagationMethod.ASM
            and spec.evanescent_policy is EvanescentPolicy.DECAY
        ):
            issues.append(
                _issue(
                    Severity.ERROR,
                    "sampling.evanescent_decay_complex_index_unsupported",
                    "The stable decay policy is currently defined only for "
                    "real refractive index.",
                    layer_id,
                    "evanescent_policy",
                    suggested_fix=(
                        "Use discard/keep or set a real refractive index."
                    ),
                )
            )

    _validate_output_grid(
        field.grid,
        output_grid,
        spec,
        issues,
        layer_id,
    )

    is_geometry_preview = bool(
        field.metadata.get("_optiprop_geometry_preview", False)
    )
    edge_fraction = (
        _edge_intensity_fraction(field)
        if inspect_edge and not is_geometry_preview
        else None
    )
    if edge_fraction is not None:
        if not math.isfinite(edge_fraction):
            issues.append(
                _issue(
                    Severity.ERROR,
                    "sampling.field_intensity_nonfinite",
                    "The field intensity contains NaN or infinite values.",
                    layer_id,
                    "field.data",
                    details={"finite": False},
                )
            )
        elif edge_fraction < 0.0:
            issues.append(
                _issue(
                    Severity.WARNING,
                    "sampling.zero_field_intensity",
                    "The input field has zero integrated intensity.",
                    layer_id,
                    "field.data",
                    details={"integrated_intensity": 0.0},
                )
            )
            edge_fraction = None
        elif (
            spec.method in _FFT_METHODS
            and spec.padding.mode is PaddingMode.NONE
            and edge_fraction > edge_warning_fraction
        ):
            issues.append(
                _issue(
                    Severity.WARNING,
                    "sampling.edge_energy_wrap_risk",
                    "Significant intensity reaches the grid edge without padding.",
                    layer_id,
                    "padding",
                    suggested_fix="Enable automatic padding or enlarge the field.",
                    details={
                        "edge_intensity_fraction": edge_fraction,
                        "warning_fraction": edge_warning_fraction,
                    },
                )
            )

    operations = _estimate_operations(
        field,
        computational_grid,
        output_grid,
        spec.method,
    )
    memory_bytes = _estimate_peak_memory_bytes(
        field,
        computational_grid,
        output_grid,
        spec,
    )
    if memory_limit_bytes is not None and memory_bytes > memory_limit_bytes:
        issues.append(
            _issue(
                Severity.ERROR,
                "sampling.memory_estimate_exceeded",
                "Estimated peak propagation memory exceeds the configured limit.",
                layer_id,
                "padding",
                suggested_fix="Reduce the grid or padding, or select another device.",
                details={
                    "estimated_bytes": memory_bytes,
                    "limit_bytes": memory_limit_bytes,
                },
            )
        )

    reported_operations: float | None = None
    if spec.method is PropagationMethod.RS_DIRECT:
        reported_operations = operations
        if operations >= rs_direct_error_operations:
            severity = Severity.ERROR
            workload = "excessive"
        elif operations >= rs_direct_warning_operations:
            severity = Severity.WARNING
            workload = "high"
        else:
            severity = Severity.INFO
            workload = "acceptable"
        issues.append(
            _issue(
                severity,
                "sampling.rs_direct_workload",
                "Estimated direct Rayleigh-Sommerfeld interaction workload.",
                layer_id,
                "method",
                suggested_fix=(
                    "Use RS-FFT or reduce the input/output grids."
                    if severity is not Severity.INFO
                    else None
                ),
                details={
                    "estimated_operations": operations,
                    "classification": workload,
                },
            )
        )

    if (
        spec.padding.mode is PaddingMode.AUTO
        and spec.distance_m != 0.0
        and computational_grid.shape != field.grid.shape
    ):
        issues.append(
            _issue(
                Severity.INFO,
                "sampling.padding_recommended",
                "Automatic padding enlarged the computational grid.",
                layer_id,
                "padding",
                details={
                    "input_shape": field.grid.shape,
                    "recommended_shape": recommended_shape,
                },
            )
        )

    fresnel_x, fresnel_y = _fresnel_numbers(
        field.grid,
        wavelength_medium_m,
        spec.distance_m,
    )
    return SamplingReport(
        method=spec.method,
        input_grid=field.grid,
        computational_grid=computational_grid,
        output_grid=output_grid,
        wavelength_medium_m=wavelength_medium_m,
        nyquist_x_per_m=1.0 / (2.0 * field.grid.dx),
        nyquist_y_per_m=1.0 / (2.0 * field.grid.dy),
        fresnel_number_x=fresnel_x,
        fresnel_number_y=fresnel_y,
        edge_intensity_fraction=edge_fraction,
        estimated_peak_memory_bytes=memory_bytes,
        estimated_operations=reported_operations,
        recommended_padded_shape=recommended_shape,
        validation=ValidationReport(tuple(issues)),
        diagnostics={
            "input_shape": (field.grid.ny, field.grid.nx),
            "computational_shape": (
                computational_grid.ny,
                computational_grid.nx,
            ),
            "output_shape": (output_grid.ny, output_grid.nx),
            "component_count": field.component_count,
            "estimated_fft_operations": (
                operations
                if spec.method is not PropagationMethod.RS_DIRECT
                else None
            ),
        },
    )


def _computational_grid(
    grid: Grid2D,
    spec: PropagationSpec,
    recommended_shape: Tuple[int, int],
    issues: list[ValidationIssue],
    layer_id: str | None,
) -> Grid2D:
    padding = spec.padding
    if padding.mode is PaddingMode.NONE:
        ny, nx = grid.ny, grid.nx
    elif padding.mode is PaddingMode.AUTO:
        ny, nx = recommended_shape
    elif padding.mode is PaddingMode.FACTOR:
        assert padding.factor is not None
        ny = math.ceil(grid.ny * padding.factor)
        nx = math.ceil(grid.nx * padding.factor)
    else:
        assert padding.shape is not None
        ny, nx = padding.shape
        if ny < grid.ny or nx < grid.nx:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "sampling.padding_explicit_too_small",
                    "Explicit padding shape cannot be smaller than the input grid.",
                    layer_id,
                    "padding.shape",
                    details={
                        "input_shape": (grid.ny, grid.nx),
                        "requested_shape": (ny, nx),
                    },
                )
            )
    return Grid2D(
        nx=nx,
        ny=ny,
        dx=grid.dx,
        dy=grid.dy,
        # Center padding places the extra sample on the high-index side when
        # an odd number of cells is added.  Preserve the original samples'
        # physical coordinates by recording the resulting half-pixel shift.
        x_center=grid.x_center
        + (((nx - grid.nx) / 2.0) - ((nx - grid.nx) // 2)) * grid.dx,
        y_center=grid.y_center
        + (((ny - grid.ny) / 2.0) - ((ny - grid.ny) // 2)) * grid.dy,
    )


def _output_grid(
    input_grid: Grid2D,
    computational_grid: Grid2D,
    wavelength_medium_m: float,
    spec: PropagationSpec,
) -> Grid2D:
    if spec.output_grid is not None:
        return spec.output_grid
    if (
        spec.method is PropagationMethod.FRESNEL_SCALED
        and spec.distance_m != 0.0
    ):
        return Grid2D(
            nx=computational_grid.nx,
            ny=computational_grid.ny,
            dx=(
                wavelength_medium_m
                * abs(spec.distance_m)
                / (computational_grid.nx * input_grid.dx)
            ),
            dy=(
                wavelength_medium_m
                * abs(spec.distance_m)
                / (computational_grid.ny * input_grid.dy)
            ),
            x_center=input_grid.x_center,
            y_center=input_grid.y_center,
        )
    return input_grid


def _validate_output_grid(
    input_grid: Grid2D,
    output_grid: Grid2D,
    spec: PropagationSpec,
    issues: list[ValidationIssue],
    layer_id: str | None,
) -> None:
    if spec.output_grid is None or spec.method not in _SAME_GRID_METHODS:
        return
    same_sampling = (
        output_grid.nx == input_grid.nx
        and output_grid.ny == input_grid.ny
        and output_grid.dx == input_grid.dx
        and output_grid.dy == input_grid.dy
        and output_grid.x_center == input_grid.x_center
        and output_grid.y_center == input_grid.y_center
    )
    if not same_sampling:
        issues.append(
            _issue(
                Severity.ERROR,
                "sampling.output_grid_unsupported",
                f"{spec.method.value} currently requires the input sampling grid.",
                layer_id,
                "output_grid",
                suggested_fix="Remove the custom output grid or use a scaled method.",
            )
        )


def _edge_intensity_fraction(field: Field2D) -> float:
    intensity = field.intensity().detach()
    if not bool(torch.isfinite(intensity).all().item()):
        return math.nan
    total = float(intensity.sum().item())
    if total == 0.0:
        return -1.0
    border = max(1, math.ceil(min(field.grid.nx, field.grid.ny) * 0.05))
    edge = intensity[:border, :].sum() + intensity[-border:, :].sum()
    if field.grid.ny > 2 * border:
        edge = edge + intensity[border:-border, :border].sum()
        edge = edge + intensity[border:-border, -border:].sum()
    return float(edge.item()) / total


def _estimate_peak_memory_bytes(
    field: Field2D,
    computational_grid: Grid2D,
    output_grid: Grid2D,
    spec: PropagationSpec,
) -> int:
    if spec.precision is PrecisionPolicy.COMPLEX128:
        item_size = 16
    elif spec.precision is PrecisionPolicy.COMPLEX64:
        item_size = 8
    else:
        item_size = field.data.element_size()
    computational_values = (
        field.component_count * computational_grid.ny * computational_grid.nx
    )
    output_values = field.component_count * output_grid.ny * output_grid.nx
    multiplier = 8 if spec.method is PropagationMethod.RS_FFT else 6
    return int(item_size * (multiplier * computational_values + output_values))


def _resolved_complex_item_size(
    field: Field2D,
    spec: PropagationSpec,
) -> int:
    if spec.precision is PrecisionPolicy.COMPLEX128:
        return 16
    if spec.precision is PrecisionPolicy.COMPLEX64:
        return 8
    return field.data.element_size()


def _maximum_evanescent_exponent(
    field: Field2D,
    wavelength_medium_m: float,
    distance_m: float,
) -> float:
    maximum_kx = math.pi / field.grid.dx
    maximum_ky = math.pi / field.grid.dy
    medium_wavenumber = 2.0 * math.pi / wavelength_medium_m
    alpha_squared = max(
        0.0,
        maximum_kx**2 + maximum_ky**2 - medium_wavenumber**2,
    )
    return math.sqrt(alpha_squared) * abs(distance_m)


def _estimate_operations(
    field: Field2D,
    computational_grid: Grid2D,
    output_grid: Grid2D,
    method: PropagationMethod,
) -> float:
    input_points = field.grid.ny * field.grid.nx
    output_points = output_grid.ny * output_grid.nx
    if method is PropagationMethod.RS_DIRECT:
        return float(field.component_count * input_points * output_points)
    points = computational_grid.ny * computational_grid.nx
    return float(
        15.0
        * field.component_count
        * points
        * math.log2(max(2, points))
    )


def _fresnel_numbers(
    grid: Grid2D,
    wavelength_medium_m: float,
    distance_m: float,
) -> tuple[float | None, float | None]:
    if distance_m == 0.0:
        return None, None
    denominator = wavelength_medium_m * abs(distance_m)
    return (
        (grid.width / 2.0) ** 2 / denominator,
        (grid.height / 2.0) ** 2 / denominator,
    )


def _issue(
    severity: Severity,
    code: str,
    message: str,
    layer_id: str | None,
    parameter_path: str | None,
    *,
    suggested_fix: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        layer_id=layer_id,
        parameter_path=parameter_path,
        suggested_fix=suggested_fix,
        details={} if details is None else details,
    )


def _validate_optional_limit(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("memory_limit_bytes must be an integer or None.")
    if value <= 0:
        raise ValueError("memory_limit_bytes must be positive.")


def _recommended_padded_shape(
    grid: Grid2D, spec: PropagationSpec
) -> Tuple[int, int]:
    if spec.distance_m == 0.0 or spec.method is PropagationMethod.RS_DIRECT:
        return grid.shape
    if spec.method in _FFT_METHODS:
        return 2 * grid.ny, 2 * grid.nx
    return grid.shape


__all__ = ["SamplingReport", "assess_sampling"]
