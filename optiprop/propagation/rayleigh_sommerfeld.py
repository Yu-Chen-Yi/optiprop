"""Canonical Rayleigh--Sommerfeld propagation.

The implementation uses the ``exp(-i omega t)`` convention and the first
Rayleigh--Sommerfeld solution (RS-I)

    h_z(dx, dy) = z / (i lambda_m R^2) * exp(i k R)
                  * (1 - 1 / (i k R)),

where ``R = sqrt(dx^2 + dy^2 + z^2)``,
``k = 2 pi n / lambda_0``, and ``lambda_m = lambda_0 / n``.  This WP-05
backend accepts only ``z > 0`` (plus the exact ``z = 0`` identity).  A
conjugated spatial Green kernel in the opposite half-space is not the exact
inverse of evanescent propagation, so negative distance is rejected until the
API has an explicit continuation policy.

``RS_FFT`` evaluates the sampled integral as a true linear Toeplitz
convolution.  The input occupies the low-index corner of an FFT buffer and
signed kernel lags are stored modulo that buffer.  Requiring an FFT shape of
at least ``(2*ny-1, 2*nx-1)`` prevents circular wrap into the requested
same-grid output block.  This buffer indexing does not change the physical
``Grid2D`` coordinates.

``RS_DIRECT`` evaluates the same sampled integral in output-point chunks and
therefore supports arbitrary regular output grids.  Both algorithms multiply
the discrete sum by the input sample area ``dx*dy``.
"""

from __future__ import annotations

import math
import numbers
import time
from typing import Any, Mapping

import torch

from ..core import (
    Field2D,
    Grid2D,
    Severity,
    ValidationError,
    ValidationIssue,
    ValidationReport,
)
from .base import (
    CancellationToken,
    PrecisionPolicy,
    PropagationMethod,
    PropagationResult,
    PropagationSpec,
    Propagator,
)
from .sampling import SamplingReport, assess_sampling


_RS_METHODS = frozenset(
    {PropagationMethod.RS_FFT, PropagationMethod.RS_DIRECT}
)
_DEFAULT_DIRECT_MEMORY_BUDGET_BYTES = 128 * 1024 * 1024
_MAX_AUTOMATIC_DIRECT_CHUNK_SIZE = 1024
_INTERACTION_MEMORY_ITEM_MULTIPLIER = 12


class RayleighSommerfeldPropagator(Propagator):
    """Exact sampled RS-I propagation by FFT or chunked direct integration."""

    def validate(
        self,
        field: Field2D,
        spec: PropagationSpec,
    ) -> ValidationReport:
        base_report = super().validate(field, spec)
        sampling = assess_sampling(field, spec)
        return base_report.merge(
            sampling.validation,
            _rayleigh_sommerfeld_validation(field, spec, sampling),
        )

    def propagate(
        self,
        field: Field2D,
        spec: PropagationSpec,
        cancel_token: CancellationToken | None = None,
    ) -> PropagationResult:
        return propagate_rayleigh_sommerfeld(
            field,
            spec,
            cancel_token=cancel_token,
        )


def propagate_rayleigh_sommerfeld(
    field: Field2D,
    spec: PropagationSpec,
    cancel_token: CancellationToken | None = None,
) -> PropagationResult:
    """Propagate ``field`` with canonical RS-FFT or RS-Direct."""

    if not isinstance(field, Field2D):
        raise TypeError("field must be a Field2D.")
    if not isinstance(spec, PropagationSpec):
        raise TypeError("spec must be a PropagationSpec.")
    if cancel_token is not None and not isinstance(
        cancel_token, CancellationToken
    ):
        raise TypeError("cancel_token must be a CancellationToken or None.")
    _check_cancelled(cancel_token)

    started = time.perf_counter()
    sampling = assess_sampling(field, spec)
    validation = sampling.validation.merge(
        _rayleigh_sommerfeld_validation(field, spec, sampling)
    )
    if not validation.is_valid:
        raise ValidationError(validation)

    target_dtype = _resolved_complex_dtype(field, spec)
    input_data = field.data.to(dtype=target_dtype)
    medium_index = spec.resolved_medium_index(field)
    output_grid = sampling.output_grid
    power_before = _integrated_intensity_float(
        input_data,
        dx=field.grid.dx,
        dy=field.grid.dy,
    )
    _check_cancelled(cancel_token)

    if spec.distance_m == 0.0:
        output_field = field.with_data(
            input_data,
            medium_index=medium_index,
        )
        elapsed = time.perf_counter() - started
        return PropagationResult(
            field=output_field,
            spec=spec,
            elapsed_s=elapsed,
            sampling=sampling,
            validation=validation,
            power_before=power_before,
            power_after=power_before,
            diagnostics={
                "identity": True,
                "algorithm": spec.method.value,
                "input_shape": field.grid.shape,
                "computational_shape": field.grid.shape,
                "output_shape": field.grid.shape,
                "dtype": str(target_dtype),
                "component_count": field.component_count,
                "medium_index_real": medium_index.real,
                "medium_index_imag": medium_index.imag,
                "power_input": power_before,
                "power_output": power_before,
                "quadrature_weight_m2": field.grid.dx * field.grid.dy,
                "kernel": "RS-I",
                "time_convention": "exp(-i*omega*t)",
            },
        )

    if spec.method is PropagationMethod.RS_FFT:
        output_data, algorithm_diagnostics = _propagate_fft(
            input_data,
            field,
            spec,
            sampling,
            medium_index,
            cancel_token,
        )
    else:
        output_data, algorithm_diagnostics = _propagate_direct(
            input_data,
            field,
            spec,
            output_grid,
            medium_index,
            cancel_token,
        )

    _check_cancelled(cancel_token)
    output_field = field.with_data(
        output_data,
        grid=output_grid,
        medium_index=medium_index,
        z_m=field.z_m + spec.distance_m,
    )
    power_after = _integrated_intensity_float(
        output_data,
        dx=output_grid.dx,
        dy=output_grid.dy,
    )
    relative_power_change = (
        (power_after - power_before) / power_before
        if power_before > 0.0
        else None
    )
    elapsed = time.perf_counter() - started
    diagnostics: Mapping[str, Any] = {
        "identity": False,
        "algorithm": spec.method.value,
        "input_shape": field.grid.shape,
        "computational_shape": sampling.computational_grid.shape,
        "output_shape": output_grid.shape,
        "dtype": str(target_dtype),
        "component_count": field.component_count,
        "vacuum_wavelength_m": field.wavelength_m,
        "medium_wavelength_real_m": (
            field.wavelength_m / medium_index.real
        ),
        "medium_index_real": medium_index.real,
        "medium_index_imag": medium_index.imag,
        "k0_per_m": 2.0 * math.pi / field.wavelength_m,
        "distance_m": spec.distance_m,
        "power_input": power_before,
        "power_output": power_after,
        "relative_power_change": relative_power_change,
        "quadrature_weight_m2": field.grid.dx * field.grid.dy,
        "kernel": "RS-I",
        "kernel_formula": (
            "z/(i*lambda_m*R^2)*exp(i*k*R)"
            "*(1-1/(i*k*R))"
        ),
        "time_convention": "exp(-i*omega*t)",
        **algorithm_diagnostics,
    }
    return PropagationResult(
        field=output_field,
        spec=spec,
        elapsed_s=elapsed,
        sampling=sampling,
        validation=validation,
        power_before=power_before,
        power_after=power_after,
        diagnostics=diagnostics,
    )


def _propagate_fft(
    input_data: torch.Tensor,
    field: Field2D,
    spec: PropagationSpec,
    sampling: SamplingReport,
    medium_index: complex,
    cancel_token: CancellationToken | None,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Evaluate a same-grid linear convolution through circulant embedding."""

    fft_ny, fft_nx = sampling.computational_grid.shape
    input_ny, input_nx = field.grid.shape
    real_dtype = _real_dtype(input_data.dtype)
    device = input_data.device

    # The low-index embedding plus modulo-addressed signed lags makes output
    # [0:ny, 0:nx] equal to the desired Toeplitz product.  No fftshift is
    # involved, which also avoids odd/even half-pixel ambiguity.
    input_buffer = torch.zeros(
        (field.component_count, fft_ny, fft_nx),
        dtype=input_data.dtype,
        device=device,
    )
    input_buffer[:, :input_ny, :input_nx] = input_data

    lag_y_index = _signed_fft_lag_indices(
        fft_ny, dtype=real_dtype, device=device
    )
    lag_x_index = _signed_fft_lag_indices(
        fft_nx, dtype=real_dtype, device=device
    )
    lag_y, lag_x = torch.meshgrid(
        lag_y_index * field.grid.dy,
        lag_x_index * field.grid.dx,
        indexing="ij",
    )
    support = (
        (torch.abs(lag_y_index).unsqueeze(1) <= input_ny - 1)
        & (torch.abs(lag_x_index).unsqueeze(0) <= input_nx - 1)
    )
    kernel = _rs_kernel(
        lag_x,
        lag_y,
        distance_m=spec.distance_m,
        wavelength_m=field.wavelength_m,
        medium_index=medium_index,
        dtype=input_data.dtype,
    )
    kernel = torch.where(support, kernel, torch.zeros_like(kernel))
    _check_cancelled(cancel_token)

    input_spectrum = torch.fft.fft2(input_buffer, dim=(-2, -1))
    _check_cancelled(cancel_token)
    kernel_spectrum = torch.fft.fft2(kernel, dim=(-2, -1))
    _check_cancelled(cancel_token)
    convolved = torch.fft.ifft2(
        input_spectrum * kernel_spectrum.unsqueeze(0),
        dim=(-2, -1),
    )
    _check_cancelled(cancel_token)
    output_data = (
        convolved[:, :input_ny, :input_nx]
        * (field.grid.dx * field.grid.dy)
    )

    expected_shape = (
        field.component_count,
        field.grid.ny,
        field.grid.nx,
    )
    if tuple(output_data.shape) != expected_shape:
        raise RuntimeError(
            "Internal RS-FFT crop produced the wrong shape: "
            f"expected {expected_shape}, got {tuple(output_data.shape)}."
        )
    return output_data, {
        "fft_shape": (fft_ny, fft_nx),
        "minimum_linear_convolution_shape": (
            2 * input_ny - 1,
            2 * input_nx - 1,
        ),
        "linear_convolution": True,
        "circulant_embedding": (
            "input at low-index origin; signed kernel lags modulo FFT shape"
        ),
        "output_slice": (
            (0, input_ny),
            (0, input_nx),
        ),
        "kernel_support_shape": (
            2 * input_ny - 1,
            2 * input_nx - 1,
        ),
        "normalization": "ifft(fft(U)*fft(h))*dx*dy",
        "arbitrary_output_grid": False,
    }


def _propagate_direct(
    input_data: torch.Tensor,
    field: Field2D,
    spec: PropagationSpec,
    output_grid: Grid2D,
    medium_index: complex,
    cancel_token: CancellationToken | None,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Evaluate the sampled RS-I integral in output-coordinate chunks."""

    chunk_size = _direct_chunk_size(spec, field)
    real_dtype = _real_dtype(input_data.dtype)
    device = input_data.device
    input_x, input_y = field.grid.meshgrid(
        dtype=real_dtype,
        device=device,
    )
    output_x, output_y = output_grid.meshgrid(
        dtype=real_dtype,
        device=device,
    )
    input_x_flat = input_x.reshape(-1)
    input_y_flat = input_y.reshape(-1)
    output_x_flat = output_x.reshape(-1)
    output_y_flat = output_y.reshape(-1)
    input_flat = input_data.reshape(field.component_count, -1)
    output_chunks: list[torch.Tensor] = []

    output_points = output_grid.ny * output_grid.nx
    chunk_count = math.ceil(output_points / chunk_size)
    peak_interactions = 0
    estimated_bytes_per_interaction = (
        _INTERACTION_MEMORY_ITEM_MULTIPLIER * input_data.element_size()
    )
    for start in range(0, output_points, chunk_size):
        _check_cancelled(cancel_token)
        stop = min(start + chunk_size, output_points)
        delta_x = (
            output_x_flat[start:stop].unsqueeze(1)
            - input_x_flat.unsqueeze(0)
        )
        delta_y = (
            output_y_flat[start:stop].unsqueeze(1)
            - input_y_flat.unsqueeze(0)
        )
        kernel = _rs_kernel(
            delta_x,
            delta_y,
            distance_m=spec.distance_m,
            wavelength_m=field.wavelength_m,
            medium_index=medium_index,
            dtype=input_data.dtype,
        )
        output_chunk = torch.matmul(input_flat, kernel.transpose(0, 1))
        output_chunks.append(
            output_chunk * (field.grid.dx * field.grid.dy)
        )
        peak_interactions = max(peak_interactions, kernel.numel())
        _check_cancelled(cancel_token)

    output_data = torch.cat(output_chunks, dim=1).reshape(
        field.component_count,
        output_grid.ny,
        output_grid.nx,
    )
    return output_data, {
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "input_point_count": field.grid.ny * field.grid.nx,
        "output_point_count": output_points,
        "peak_chunk_interactions": peak_interactions,
        "estimated_peak_interaction_bytes": (
            peak_interactions * estimated_bytes_per_interaction
        ),
        "chunk_selection": (
            "explicit"
            if "chunk_size" in spec.options
            else "automatic_memory_budget"
        ),
        "direct_memory_budget_bytes": _direct_memory_budget(spec),
        "normalization": "sum(U*h)*dx*dy",
        "arbitrary_output_grid": True,
    }


def _rs_kernel(
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    *,
    distance_m: float,
    wavelength_m: float,
    medium_index: complex,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the forward, outgoing exact RS-I impulse response."""

    if distance_m <= 0.0:
        raise ValueError("The outgoing RS-I kernel requires positive distance.")
    real_dtype = _real_dtype(dtype)
    distance = torch.as_tensor(
        distance_m,
        dtype=real_dtype,
        device=delta_x.device,
    )
    radius = torch.sqrt(
        delta_x.square() + delta_y.square() + distance.square()
    )
    k = torch.as_tensor(
        2.0 * math.pi * medium_index / wavelength_m,
        dtype=dtype,
        device=delta_x.device,
    )
    lambda_medium = torch.as_tensor(
        wavelength_m / medium_index,
        dtype=dtype,
        device=delta_x.device,
    )
    kr = k * radius.to(dtype)
    return (
        distance.to(dtype)
        / (1j * lambda_medium * radius.square().to(dtype))
        * torch.exp(1j * kr)
        * (1.0 - 1.0 / (1j * kr))
    )


def _rayleigh_sommerfeld_validation(
    field: Field2D,
    spec: PropagationSpec,
    sampling: SamplingReport,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if spec.method not in _RS_METHODS:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="propagation.unsupported_method",
                message=(
                    "RayleighSommerfeldPropagator only supports rs_fft and "
                    "rs_direct."
                ),
                parameter_path="method",
                suggested_fix="Select an RS propagation method.",
                details={"requested_method": spec.method.value},
            )
        )
        return ValidationReport(tuple(issues))

    if (
        spec.distance_m == 0.0
        and spec.output_grid is not None
        and spec.output_grid != field.grid
    ):
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="rayleigh_sommerfeld.zero_distance_output_grid",
                message=(
                    "Zero-distance RS propagation is an identity and cannot "
                    "resample onto a different grid."
                ),
                parameter_path="output_grid",
                suggested_fix="Remove output_grid or use the exact input grid.",
            )
        )

    if spec.distance_m < 0.0:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="rs.negative_distance_unsupported",
                message=(
                    "Canonical RS-I currently supports forward propagation "
                    "only; a negative-distance spatial kernel is not an exact "
                    "evanescent inverse."
                ),
                parameter_path="distance_m",
                suggested_fix=(
                    "Use a positive distance, or use ASM with an explicit "
                    "evanescent continuation policy for backpropagation."
                ),
                details={"distance_m": spec.distance_m},
            )
        )

    if spec.method is PropagationMethod.RS_FFT and spec.distance_m != 0.0:
        minimum_shape = (
            2 * field.grid.ny - 1,
            2 * field.grid.nx - 1,
        )
        actual_shape = sampling.computational_grid.shape
        if (
            actual_shape[0] < minimum_shape[0]
            or actual_shape[1] < minimum_shape[1]
        ):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="rayleigh_sommerfeld.fft_padding_too_small",
                    message=(
                        "RS-FFT requires a convolution buffer of at least "
                        "(2*ny-1, 2*nx-1) to prevent circular wrap."
                    ),
                    parameter_path="padding",
                    suggested_fix=(
                        "Use automatic padding or provide a sufficiently "
                        "large explicit/factor padding."
                    ),
                    details={
                        "input_shape": field.grid.shape,
                        "minimum_shape": minimum_shape,
                        "actual_shape": actual_shape,
                    },
                )
            )

    if spec.method is PropagationMethod.RS_DIRECT:
        memory_budget_valid = True
        try:
            _direct_memory_budget(spec)
        except (TypeError, ValueError) as exc:
            memory_budget_valid = False
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="rayleigh_sommerfeld.invalid_memory_budget",
                    message=str(exc),
                    parameter_path="options.direct_memory_budget_bytes",
                    suggested_fix=(
                        "Set direct_memory_budget_bytes to a positive integer."
                    ),
                )
            )
        if "chunk_size" in spec.options or memory_budget_valid:
            try:
                _direct_chunk_size(spec, field)
            except (TypeError, ValueError) as exc:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="rayleigh_sommerfeld.invalid_chunk_size",
                        message=str(exc),
                        parameter_path="options.chunk_size",
                        suggested_fix="Set chunk_size to a positive integer.",
                    )
                )
    if spec.distance_m > 0.0:
        issues.extend(
            _rs_kernel_sampling_issues(
                field,
                sampling.output_grid,
                spec,
            )
        )
    return ValidationReport(tuple(issues))


def _rs_kernel_sampling_issues(
    field: Field2D,
    output_grid: Grid2D,
    spec: PropagationSpec,
) -> list[ValidationIssue]:
    """Conservatively flag grids that undersample the RS kernel phase."""

    input_xmin, input_xmax, input_ymin, input_ymax = (
        field.grid.sample_extent
    )
    output_xmin, output_xmax, output_ymin, output_ymax = (
        output_grid.sample_extent
    )
    maximum_delta_x = max(
        abs(output_xmin - input_xmax),
        abs(output_xmax - input_xmin),
    )
    maximum_delta_y = max(
        abs(output_ymin - input_ymax),
        abs(output_ymax - input_ymin),
    )
    wavelength_medium = (
        field.wavelength_m / spec.resolved_medium_index(field).real
    )
    distance = spec.distance_m
    maximum_frequency_x = maximum_delta_x / (
        wavelength_medium
        * math.sqrt(distance**2 + maximum_delta_x**2)
    )
    maximum_frequency_y = maximum_delta_y / (
        wavelength_medium
        * math.sqrt(distance**2 + maximum_delta_y**2)
    )
    nyquist_x = min(
        1.0 / (2.0 * field.grid.dx),
        1.0 / (2.0 * output_grid.dx),
    )
    nyquist_y = min(
        1.0 / (2.0 * field.grid.dy),
        1.0 / (2.0 * output_grid.dy),
    )
    findings: list[ValidationIssue] = []
    for axis, maximum_frequency, nyquist, maximum_delta in (
        ("x", maximum_frequency_x, nyquist_x, maximum_delta_x),
        ("y", maximum_frequency_y, nyquist_y, maximum_delta_y),
    ):
        if maximum_frequency <= nyquist:
            continue
        findings.append(
            ValidationIssue(
                severity=Severity.WARNING,
                code=f"rayleigh_sommerfeld.kernel_sampling_{axis}",
                message=(
                    f"The {axis.upper()} sampling may not resolve the "
                    "fastest local phase variation of the RS-I kernel."
                ),
                parameter_path=f"output_grid.d{axis}",
                suggested_fix=(
                    f"Reduce input/output d{axis}, reduce the observation "
                    "extent, or increase propagation distance."
                ),
                details={
                    "axis": axis,
                    "maximum_kernel_frequency_per_m": maximum_frequency,
                    "limiting_nyquist_per_m": nyquist,
                    "maximum_coordinate_difference_m": maximum_delta,
                    "wavelength_medium_m": wavelength_medium,
                },
            )
        )
    return findings


def _direct_chunk_size(spec: PropagationSpec, field: Field2D) -> int:
    if "chunk_size" not in spec.options:
        budget = _direct_memory_budget(spec)
        input_points = field.grid.ny * field.grid.nx
        bytes_per_interaction = (
            _INTERACTION_MEMORY_ITEM_MULTIPLIER
            * _resolved_complex_item_size(field, spec)
        )
        affordable = budget // max(1, input_points * bytes_per_interaction)
        return max(
            1,
            min(_MAX_AUTOMATIC_DIRECT_CHUNK_SIZE, int(affordable)),
        )

    value = spec.options["chunk_size"]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("RS-Direct chunk_size must be a positive integer.")
    chunk_size = int(value)
    if chunk_size <= 0:
        raise ValueError("RS-Direct chunk_size must be a positive integer.")
    return chunk_size


def _direct_memory_budget(spec: PropagationSpec) -> int:
    value = spec.options.get(
        "direct_memory_budget_bytes",
        _DEFAULT_DIRECT_MEMORY_BUDGET_BYTES,
    )
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            "RS-Direct direct_memory_budget_bytes must be a positive integer."
        )
    budget = int(value)
    if budget <= 0:
        raise ValueError(
            "RS-Direct direct_memory_budget_bytes must be a positive integer."
        )
    return budget


def _signed_fft_lag_indices(
    size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(size, dtype=dtype, device=device)
    last_positive = (size - 1) // 2
    return torch.where(indices <= last_positive, indices, indices - size)


def _resolved_complex_dtype(
    field: Field2D,
    spec: PropagationSpec,
) -> torch.dtype:
    if spec.precision is PrecisionPolicy.COMPLEX64:
        return torch.complex64
    if spec.precision is PrecisionPolicy.COMPLEX128:
        return torch.complex128
    return field.data.dtype


def _resolved_complex_item_size(
    field: Field2D,
    spec: PropagationSpec,
) -> int:
    if spec.precision is PrecisionPolicy.COMPLEX128:
        return 16
    if spec.precision is PrecisionPolicy.COMPLEX64:
        return 8
    return field.data.element_size()


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype is torch.complex64 else torch.float64


def _integrated_intensity_float(
    data: torch.Tensor,
    *,
    dx: float,
    dy: float,
) -> float:
    intensity = torch.sum(torch.abs(data.detach()) ** 2)
    return float((intensity * dx * dy).item())


def _check_cancelled(cancel_token: CancellationToken | None) -> None:
    if cancel_token is not None:
        cancel_token.throw_if_cancelled()


__all__ = [
    "RayleighSommerfeldPropagator",
    "propagate_rayleigh_sommerfeld",
]
