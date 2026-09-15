"""Canonical paraxial Fresnel propagation backends.

The implementation uses the ``exp(-i omega t)`` convention, so an outgoing
plane wave propagating in positive Z carries the global phase
``exp(+i k z)``.

Two complementary algorithms are provided:

``FRESNEL_TF``
    A same-sampling transfer-function method,

    ``H(fx, fy) = exp(i k z) exp(-i z (kx**2 + ky**2) / (2 k))``.

    The input may be zero padded to reduce circular-convolution wraparound;
    the propagated computational window is then cropped back to the exact
    input grid.

``FRESNEL_SCALED``
    The single-FFT Fresnel transform,

    ``U2 = k exp(i k z)/(i 2 pi z) * Q2 * FT{U1 Q1}``.

    Its output sampling is constrained by the DFT:
    ``dx2 = lambda_medium |z|/(Nx dx1)`` and equivalently in Y.  Padding
    increases the output sample count; unlike the transfer-function method,
    the scaled result is not cropped.  Output-grid translations are supported
    by exact DFT phase ramps, while arbitrary magnification is intentionally
    left to a future shifted/scaled Fresnel transform.

Both methods operate component-wise on canonical ``[C, ny, nx]`` tensors and
preserve PyTorch device placement and autograd graphs.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Tuple

import torch
import torch.nn.functional as torch_functional

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


_FRESNEL_METHODS = frozenset(
    {
        PropagationMethod.FRESNEL_TF,
        PropagationMethod.FRESNEL_SCALED,
    }
)


class FresnelPropagator(Propagator):
    """Paraxial same-grid and natural-sampling Fresnel propagator."""

    def validate(
        self,
        field: Field2D,
        spec: PropagationSpec,
    ) -> ValidationReport:
        base_report = super().validate(field, spec)
        sampling = assess_sampling(field, spec)
        return base_report.merge(
            sampling.validation,
            _fresnel_validation(field, spec, sampling),
        )

    def propagate(
        self,
        field: Field2D,
        spec: PropagationSpec,
        cancel_token: CancellationToken | None = None,
    ) -> PropagationResult:
        return propagate_fresnel(field, spec, cancel_token=cancel_token)


def propagate_fresnel(
    field: Field2D,
    spec: PropagationSpec,
    cancel_token: CancellationToken | None = None,
) -> PropagationResult:
    """Propagate a canonical field with a selected Fresnel algorithm."""

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
        _fresnel_validation(field, spec, sampling)
    )
    if not validation.is_valid:
        raise ValidationError(validation)

    target_dtype = _resolved_complex_dtype(field, spec)
    input_data = field.data.to(dtype=target_dtype)
    medium_index = spec.resolved_medium_index(field)
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
                "padding": (0, 0, 0, 0),
                "dtype": str(target_dtype),
                "power_input": power_before,
                "power_output_full_computational": power_before,
                "power_output": power_before,
                "crop_retained_power_fraction": (
                    1.0 if power_before > 0.0 else None
                ),
                "global_phase_included": True,
            },
        )

    if spec.method is PropagationMethod.FRESNEL_TF:
        output_data, output_grid, algorithm_diagnostics = (
            _propagate_transfer_function(
                input_data,
                field,
                spec,
                sampling,
                medium_index,
                cancel_token,
            )
        )
    else:
        output_data, output_grid, algorithm_diagnostics = (
            _propagate_single_fft(
                input_data,
                field,
                spec,
                sampling,
                medium_index,
                cancel_token,
            )
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
        "medium_index_real": medium_index.real,
        "medium_index_imag": medium_index.imag,
        "k0_per_m": 2.0 * math.pi / field.wavelength_m,
        "distance_m": spec.distance_m,
        "power_input": power_before,
        "power_output": power_after,
        "relative_power_change": relative_power_change,
        "global_phase_included": True,
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


def _propagate_transfer_function(
    input_data: torch.Tensor,
    field: Field2D,
    spec: PropagationSpec,
    sampling: SamplingReport,
    medium_index: complex,
    cancel_token: CancellationToken | None,
) -> tuple[torch.Tensor, Grid2D, Mapping[str, Any]]:
    computational_shape = sampling.computational_grid.shape
    padded, crop_slices, padding = _pad_to_shape(
        input_data,
        computational_shape,
    )
    _check_cancelled(cancel_token)

    target_dtype = input_data.dtype
    real_dtype = _real_dtype(target_dtype)
    device = input_data.device
    ny_fft, nx_fft = computational_shape
    frequency_x = torch.fft.fftfreq(
        nx_fft,
        d=field.grid.dx,
        dtype=real_dtype,
        device=device,
    )
    frequency_y = torch.fft.fftfreq(
        ny_fft,
        d=field.grid.dy,
        dtype=real_dtype,
        device=device,
    )
    fy, fx = torch.meshgrid(frequency_y, frequency_x, indexing="ij")
    transverse_k_squared = (2.0 * math.pi) ** 2 * (
        fx.square() + fy.square()
    )
    k0 = 2.0 * math.pi / field.wavelength_m
    k = torch.as_tensor(
        k0 * medium_index,
        dtype=target_dtype,
        device=device,
    )
    transfer = torch.exp(
        1j * k * spec.distance_m
        - 1j
        * transverse_k_squared.to(target_dtype)
        * spec.distance_m
        / (2.0 * k)
    )

    spectrum = torch.fft.fft2(padded, dim=(-2, -1))
    _check_cancelled(cancel_token)
    propagated_padded = torch.fft.ifft2(
        spectrum * transfer.unsqueeze(0),
        dim=(-2, -1),
    )
    _check_cancelled(cancel_token)
    output_data = propagated_padded[crop_slices]
    if tuple(output_data.shape) != tuple(field.data.shape):
        raise RuntimeError(
            "Internal Fresnel-TF crop did not restore canonical field shape: "
            f"expected {tuple(field.data.shape)}, got {tuple(output_data.shape)}."
        )

    power_full = _integrated_intensity_float(
        propagated_padded,
        dx=field.grid.dx,
        dy=field.grid.dy,
    )
    power_cropped = _integrated_intensity_float(
        output_data,
        dx=field.grid.dx,
        dy=field.grid.dy,
    )
    crop_fraction = (
        power_cropped / power_full if power_full > 0.0 else None
    )
    return (
        output_data,
        field.grid,
        {
            "padding": padding,
            "transfer_function": (
                "exp(i*k*z)*exp(-i*z*(kx^2+ky^2)/(2*k))"
            ),
            "power_output_full_computational": power_full,
            "power_output_cropped": power_cropped,
            "crop_retained_power_fraction": crop_fraction,
            "natural_output_sampling": False,
        },
    )


def _propagate_single_fft(
    input_data: torch.Tensor,
    field: Field2D,
    spec: PropagationSpec,
    sampling: SamplingReport,
    medium_index: complex,
    cancel_token: CancellationToken | None,
) -> tuple[torch.Tensor, Grid2D, Mapping[str, Any]]:
    computational_grid = sampling.computational_grid
    output_grid = sampling.output_grid
    padded, _crop_slices, padding = _pad_to_shape(
        input_data,
        computational_grid.shape,
    )
    _check_cancelled(cancel_token)

    target_dtype = input_data.dtype
    real_dtype = _real_dtype(target_dtype)
    device = input_data.device
    x1, y1 = computational_grid.meshgrid(
        dtype=real_dtype,
        device=device,
    )
    x2, y2 = output_grid.meshgrid(dtype=real_dtype, device=device)

    k0 = 2.0 * math.pi / field.wavelength_m
    # Scaled Fresnel validation requires a real refractive index because the
    # FFT samples a real spatial-frequency axis.
    k_value = k0 * medium_index.real
    k = torch.as_tensor(k_value, dtype=real_dtype, device=device)
    distance = torch.as_tensor(
        spec.distance_m,
        dtype=real_dtype,
        device=device,
    )
    input_chirp = torch.exp(
        (1j * k * (x1.square() + y1.square()) / (2.0 * distance)).to(
            target_dtype
        )
    )
    chirped_input = padded * input_chirp.unsqueeze(0)

    # Evaluate the continuous Fourier transform at the exact output-grid
    # frequencies.  The phase ramps account for arbitrary physical grid
    # centres and for Grid2D's geometric-centre convention on even arrays.
    wavelength_medium = field.wavelength_m / medium_index.real
    frequency_x = x2[0, :] / (wavelength_medium * spec.distance_m)
    frequency_y = y2[:, 0] / (wavelength_medium * spec.distance_m)
    x_start = x1[0, 0]
    y_start = y1[0, 0]
    frequency_x_start = frequency_x[0]
    frequency_y_start = frequency_y[0]
    relative_x = x1[0, :] - x_start
    relative_y = y1[:, 0] - y_start
    input_frequency_ramp = torch.exp(
        (
            -1j
            * 2.0
            * math.pi
            * (
                relative_y.unsqueeze(1) * frequency_y_start
                + relative_x.unsqueeze(0) * frequency_x_start
            )
        ).to(target_dtype)
    )
    transform_input = chirped_input * input_frequency_ramp.unsqueeze(0)

    if spec.distance_m > 0.0:
        transformed = torch.fft.fft2(transform_input, dim=(-2, -1))
        transform_direction = "fft"
    else:
        transformed = torch.fft.ifft2(transform_input, dim=(-2, -1))
        transformed = transformed * (
            computational_grid.nx * computational_grid.ny
        )
        transform_direction = "ifft_unnormalized"
    _check_cancelled(cancel_token)

    output_origin_ramp = torch.exp(
        (
            -1j
            * 2.0
            * math.pi
            * (
                y_start * frequency_y.unsqueeze(1)
                + x_start * frequency_x.unsqueeze(0)
            )
        ).to(target_dtype)
    )
    continuous_transform = (
        transformed
        * output_origin_ramp.unsqueeze(0)
        * (computational_grid.dx * computational_grid.dy)
    )
    output_chirp = torch.exp(
        (1j * k * (x2.square() + y2.square()) / (2.0 * distance)).to(
            target_dtype
        )
    )
    prefactor = torch.exp((1j * k * distance).to(target_dtype)) * (
        k.to(target_dtype)
        / (1j * 2.0 * math.pi * distance.to(target_dtype))
    )
    output_data = (
        prefactor
        * output_chirp.unsqueeze(0)
        * continuous_transform
    )
    _check_cancelled(cancel_token)

    expected_shape = (
        field.component_count,
        output_grid.ny,
        output_grid.nx,
    )
    if tuple(output_data.shape) != expected_shape:
        raise RuntimeError(
            "Internal Fresnel-scaled transform produced the wrong shape: "
            f"expected {expected_shape}, got {tuple(output_data.shape)}."
        )
    return (
        output_data,
        output_grid,
        {
            "padding": padding,
            "transform_direction": transform_direction,
            "transform_normalization": (
                "continuous_fft_times_dx_dy"
            ),
            "natural_output_sampling": True,
            "output_dx_m": output_grid.dx,
            "output_dy_m": output_grid.dy,
            "output_x_center_m": output_grid.x_center,
            "output_y_center_m": output_grid.y_center,
            "medium_wavelength_m": wavelength_medium,
            "single_fft_prefactor": "k*exp(i*k*z)/(i*2*pi*z)",
            "crop_retained_power_fraction": None,
        },
    )


def _fresnel_validation(
    field: Field2D,
    spec: PropagationSpec,
    sampling: SamplingReport,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if spec.method not in _FRESNEL_METHODS:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="propagation.unsupported_method",
                message=(
                    "FresnelPropagator only supports fresnel_tf and "
                    "fresnel_scaled."
                ),
                parameter_path="method",
                suggested_fix="Select a Fresnel propagation method.",
                details={"requested_method": spec.method.value},
            )
        )
        return ValidationReport(tuple(issues))

    if spec.distance_m == 0.0 and spec.output_grid is not None:
        if spec.output_grid != field.grid:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="fresnel.zero_distance_output_grid",
                    message=(
                        "Zero-distance Fresnel propagation is an identity and "
                        "cannot resample onto a different grid."
                    ),
                    parameter_path="output_grid",
                    suggested_fix=(
                        "Remove output_grid or use the exact input grid."
                    ),
                )
            )

    if spec.method is PropagationMethod.FRESNEL_SCALED:
        medium_index = spec.resolved_medium_index(field)
        if medium_index.imag != 0.0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="fresnel.scaled_complex_index_unsupported",
                    message=(
                        "The single-FFT scaled Fresnel method requires a real "
                        "refractive index because its Fourier sampling axis "
                        "must be real."
                    ),
                    parameter_path="medium_index",
                    suggested_fix=(
                        "Use Fresnel-TF for a lossy medium or set Im(n)=0."
                    ),
                    details={
                        "medium_index_real": medium_index.real,
                        "medium_index_imag": medium_index.imag,
                    },
                )
            )
        if spec.distance_m != 0.0:
            expected = _natural_scaled_grid(field, spec, sampling)
            actual = sampling.output_grid
            if (
                actual.shape != expected.shape
                or not _close(actual.dx, expected.dx)
                or not _close(actual.dy, expected.dy)
            ):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="fresnel.scaled_output_sampling",
                        message=(
                            "Single-FFT Fresnel propagation requires its "
                            "natural DFT output shape and sampling."
                        ),
                        parameter_path="output_grid",
                        suggested_fix=(
                            "Remove output_grid or match the reported natural "
                            "nx, ny, dx and dy. Grid centre translations are "
                            "supported."
                        ),
                        details={
                            "expected_shape": expected.shape,
                            "expected_dx_m": expected.dx,
                            "expected_dy_m": expected.dy,
                            "actual_shape": actual.shape,
                            "actual_dx_m": actual.dx,
                            "actual_dy_m": actual.dy,
                        },
                    )
                )
    return ValidationReport(tuple(issues))


def _natural_scaled_grid(
    field: Field2D,
    spec: PropagationSpec,
    sampling: SamplingReport,
) -> Grid2D:
    medium_index = spec.resolved_medium_index(field)
    wavelength_medium = field.wavelength_m / medium_index.real
    computational = sampling.computational_grid
    return Grid2D(
        nx=computational.nx,
        ny=computational.ny,
        dx=(
            wavelength_medium
            * abs(spec.distance_m)
            / (computational.nx * field.grid.dx)
        ),
        dy=(
            wavelength_medium
            * abs(spec.distance_m)
            / (computational.ny * field.grid.dy)
        ),
        x_center=field.grid.x_center,
        y_center=field.grid.y_center,
    )


def _resolved_complex_dtype(
    field: Field2D,
    spec: PropagationSpec,
) -> torch.dtype:
    if spec.precision is PrecisionPolicy.COMPLEX64:
        return torch.complex64
    if spec.precision is PrecisionPolicy.COMPLEX128:
        return torch.complex128
    return field.data.dtype


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype is torch.complex64 else torch.float64


def _pad_to_shape(
    data: torch.Tensor,
    target_shape: Tuple[int, int],
) -> tuple[
    torch.Tensor,
    tuple[slice, slice, slice],
    tuple[int, int, int, int],
]:
    input_ny, input_nx = data.shape[-2:]
    target_ny, target_nx = target_shape
    if target_ny < input_ny or target_nx < input_nx:
        raise ValueError(
            "Computational shape cannot be smaller than the input shape."
        )
    total_y = target_ny - input_ny
    total_x = target_nx - input_nx
    top = total_y // 2
    bottom = total_y - top
    left = total_x // 2
    right = total_x - left
    if total_y == 0 and total_x == 0:
        padded = data
    else:
        padded = torch_functional.pad(
            data,
            (left, right, top, bottom),
            mode="constant",
            value=0.0,
        )
    return (
        padded,
        (
            slice(None),
            slice(top, top + input_ny),
            slice(left, left + input_nx),
        ),
        (left, right, top, bottom),
    )


def _integrated_intensity_float(
    data: torch.Tensor,
    *,
    dx: float,
    dy: float,
) -> float:
    intensity = torch.sum(torch.abs(data.detach()) ** 2)
    return float((intensity * dx * dy).item())


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=0.0)


def _check_cancelled(cancel_token: CancellationToken | None) -> None:
    if cancel_token is not None:
        cancel_token.throw_if_cancelled()


__all__ = ["FresnelPropagator", "propagate_fresnel"]
