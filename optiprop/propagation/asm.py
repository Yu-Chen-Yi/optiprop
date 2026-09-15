"""Canonical Angular Spectrum Method (ASM) propagators.

The implementation uses the ``exp(-i omega t)`` convention.  A positive-Z
outgoing plane wave therefore advances as ``exp(+i kz z)``:

    k0 = 2 pi / wavelength_0
    kz = sqrt((k0 n)^2 - kx^2 - ky^2)
    H(kx, ky; z) = exp(i kz z)

The square-root radiation branch has positive real part; on the imaginary
axis it has non-negative imaginary part.  For passive media this gives both
``Re(kz) >= 0`` and ``Im(kz) >= 0``.

BLAS applies the band limits derived by Matsushima and Shimobaba,
"Band-limited angular spectrum method for numerical simulation of free-space
propagation in far and near fields", Optics Express 17(22), 19662-19673
(2009), DOI 10.1364/OE.17.019662.  For a centred computational window of
extent Lx by Ly, the limits used here are:

    |fx| <= 1 / (lambda_m sqrt(1 + (2 |z| / Lx)^2))
    |fy| <= 1 / (lambda_m sqrt(1 + (2 |z| / Ly)^2))

The two-dimensional support is the intersection

    fx^2 / fx_limit^2 + (lambda_m fy)^2 <= 1
    (lambda_m fx)^2 + fy^2 / fy_limit^2 <= 1

with the propagating disk and the sampled FFT domain.  This is the centred,
aligned input/output-window case; shifted windows and scaled output sampling
require a different formulation and are rejected by the shared sampling
validation.

For real-index media, ``DECAY`` applies ``exp(-alpha |z|)`` to evanescent
components as a stable regularized continuation. ``KEEP`` applies the exact
``exp(i kz z)`` transfer and preflight reports or blocks ill-conditioned
backward amplification. Under this time convention passive loss uses
``Im(n) >= 0``; active media and backward propagation through loss are
rejected.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Mapping, Tuple

import torch
import torch.nn.functional as torch_functional

from ..core import (
    Field2D,
    Severity,
    ValidationError,
    ValidationIssue,
    ValidationReport,
)
from .base import (
    CancellationToken,
    EvanescentPolicy,
    PrecisionPolicy,
    PropagationMethod,
    PropagationResult,
    PropagationSpec,
    Propagator,
)
from .sampling import SamplingReport, assess_sampling


_ASM_METHODS = frozenset({PropagationMethod.ASM, PropagationMethod.BLAS})


class AngularSpectrumPropagator(Propagator):
    """Exact scalar/component-wise ASM and centred Matsushima BLAS."""

    def validate(
        self,
        field: Field2D,
        spec: PropagationSpec,
    ) -> ValidationReport:
        base_report = super().validate(field, spec)
        sampling_report = assess_sampling(field, spec)
        if spec.method in _ASM_METHODS:
            return base_report.merge(sampling_report.validation)
        issue = ValidationIssue(
            severity=Severity.ERROR,
            code="propagation.unsupported_method",
            message="AngularSpectrumPropagator only supports ASM and BLAS.",
            parameter_path="method",
            suggested_fix="Select 'asm' or 'blas'.",
            details={"requested_method": spec.method.value},
        )
        return base_report.merge(
            sampling_report.validation,
            ValidationReport((issue,)),
        )

    def propagate(
        self,
        field: Field2D,
        spec: PropagationSpec,
        cancel_token: CancellationToken | None = None,
    ) -> PropagationResult:
        return propagate_angular_spectrum(
            field,
            spec,
            cancel_token=cancel_token,
        )


def propagate_angular_spectrum(
    field: Field2D,
    spec: PropagationSpec,
    cancel_token: CancellationToken | None = None,
    *,
    window_shift_m: tuple[float, float] = (0.0, 0.0),
) -> PropagationResult:
    """Propagate a canonical ``Field2D`` with ASM or centred-window BLAS.

    Both field components are transformed together over their final two
    dimensions.  Padding is zero-filled and the inverse result is cropped with
    the exact inverse slices, preserving the input grid centre and canonical
    output shape.
    """

    if not isinstance(field, Field2D):
        raise TypeError("field must be a Field2D.")
    if not isinstance(spec, PropagationSpec):
        raise TypeError("spec must be a PropagationSpec.")
    if cancel_token is not None and not isinstance(
        cancel_token, CancellationToken
    ):
        raise TypeError("cancel_token must be a CancellationToken or None.")
    _check_cancelled(cancel_token)

    shift_x, shift_y = (float(v) for v in window_shift_m)
    if not all(math.isfinite(v) for v in (shift_x, shift_y)):
        raise ValueError('Window shifts must be finite metres.')
    shifted = shift_x != 0.0 or shift_y != 0.0
    if shifted and spec.method is not PropagationMethod.ASM:
        raise ValueError('Moving windows support ASM only, not centred BLAS.')
    started = time.perf_counter()
    sampling = assess_sampling(field, spec)
    validation = sampling.validation
    if spec.method not in _ASM_METHODS:
        validation = validation.with_issue(
            ValidationIssue(
                severity=Severity.ERROR,
                code="propagation.unsupported_method",
                message="Angular-spectrum propagation requires ASM or BLAS.",
                parameter_path="method",
                suggested_fix="Select 'asm' or 'blas'.",
                details={"requested_method": spec.method.value},
            )
        )
    if not validation.is_valid:
        raise ValidationError(validation)

    _check_cancelled(cancel_token)
    target_dtype = _resolved_complex_dtype(field, spec)
    input_data = field.data.to(dtype=target_dtype)
    medium_index = spec.resolved_medium_index(field)
    power_before = _integrated_intensity_float(input_data, field)

    # z=0 is an identity and must not accidentally apply the BLAS or
    # evanescent masks.  Precision and explicit medium context are still
    # reflected in the returned immutable field.
    if spec.distance_m == 0.0 and not shifted:
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
            power_after=_integrated_intensity_float(input_data, output_field),
            diagnostics={
                "identity": True,
                "input_shape": field.grid.shape,
                "computational_shape": field.grid.shape,
                "padding": (0, 0, 0, 0),
                "dtype": str(target_dtype),
                "medium_index_real": medium_index.real,
                "medium_index_imag": medium_index.imag,
                "power_input": power_before,
                "power_output_full_computational": power_before,
                "power_output_cropped": power_before,
                "crop_retained_power_fraction": (
                    1.0 if power_before > 0.0 else None
                ),
                "relative_full_power_change": (
                    0.0 if power_before > 0.0 else None
                ),
            },
        )

    computational_shape = sampling.computational_grid.shape
    padded, crop_slices, padding = _pad_to_shape(
        input_data,
        computational_shape,
    )
    _check_cancelled(cancel_token)

    ny_fft, nx_fft = computational_shape
    real_dtype = (
        torch.float32 if target_dtype == torch.complex64 else torch.float64
    )
    device = padded.device
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
    ky, kx = torch.meshgrid(
        2.0 * math.pi * frequency_y,
        2.0 * math.pi * frequency_x,
        indexing="ij",
    )
    transverse_k_squared = kx.square() + ky.square()

    k0 = 2.0 * math.pi / field.wavelength_m
    k = torch.as_tensor(
        k0 * medium_index,
        dtype=target_dtype,
        device=device,
    )
    kz = torch.sqrt(k.square() - transverse_k_squared.to(target_dtype))
    kz = _radiation_branch(kz, k0=k0, medium_index=medium_index)

    # For complex media the propagating/evanescent split is defined from the
    # real-index phase wavelength.  The exact complex k still controls H.
    propagating_limit = k0 * medium_index.real
    propagating_mask = transverse_k_squared <= propagating_limit**2

    transfer_mask = torch.ones_like(propagating_mask)
    cutoff_x_per_m: float | None = None
    cutoff_y_per_m: float | None = None
    if spec.method is PropagationMethod.BLAS:
        wavelength_medium = field.wavelength_m / medium_index.real
        theoretical_cutoff_x, theoretical_cutoff_y = _matsushima_cutoffs(
            wavelength_medium_m=wavelength_medium,
            distance_m=spec.distance_m,
            width_m=nx_fft * field.grid.dx,
            height_m=ny_fft * field.grid.dy,
        )
        # The analytic BLAS boundary is intersected with the sampled FFT
        # domain.  Report the effective cutoff that is actually applied.
        cutoff_x_per_m = min(
            theoretical_cutoff_x,
            1.0 / (2.0 * field.grid.dx),
        )
        cutoff_y_per_m = min(
            theoretical_cutoff_y,
            1.0 / (2.0 * field.grid.dy),
        )
        frequency_x_grid = frequency_x.unsqueeze(0)
        frequency_y_grid = frequency_y.unsqueeze(1)
        matsushima_x = (
            (frequency_x_grid / theoretical_cutoff_x).square()
            + (wavelength_medium * frequency_y_grid).square()
            <= 1.0
        )
        matsushima_y = (
            (wavelength_medium * frequency_x_grid).square()
            + (frequency_y_grid / theoretical_cutoff_y).square()
            <= 1.0
        )
        transfer_mask = matsushima_x & matsushima_y & propagating_mask
    elif spec.evanescent_policy is EvanescentPolicy.DISCARD:
        transfer_mask = propagating_mask

    # Avoid computing an exponentially growing value in bins that will be
    # discarded (important for backward propagation with DISCARD).
    if (
        spec.method is PropagationMethod.BLAS
        or spec.evanescent_policy is EvanescentPolicy.DISCARD
    ):
        safe_kz = torch.where(transfer_mask, kz, torch.zeros_like(kz))
        transfer = torch.exp(1j * safe_kz * spec.distance_m)
        transfer = transfer * transfer_mask.to(dtype=target_dtype)
    elif spec.evanescent_policy is EvanescentPolicy.DECAY:
        propagating_transfer = torch.exp(1j * kz * spec.distance_m)
        stable_evanescent_transfer = torch.exp(
            -torch.abs(kz.imag * spec.distance_m)
        ).to(dtype=target_dtype)
        transfer = torch.where(
            propagating_mask,
            propagating_transfer,
            stable_evanescent_transfer,
        )
    else:
        # KEEP is the exact continuation. Shared validation warns or rejects
        # ill-conditioned backward use according to the maximum exponent.
        transfer = torch.exp(1j * kz * spec.distance_m)

    if spec.distance_m == 0.0:
        transfer = torch.ones_like(transfer)
        transfer_mask = torch.ones_like(transfer_mask)
    # Positive sign samples E(x + shift), rather than translating the beam.
    transfer = transfer * torch.exp(1j * (kx * shift_x + ky * shift_y))
    _check_cancelled(cancel_token)
    spectrum = torch.fft.fft2(padded, dim=(-2, -1))
    _check_cancelled(cancel_token)
    removed_spectral_fraction = _removed_spectral_energy_fraction(
        spectrum,
        transfer_mask,
    )
    propagated_spectrum = spectrum * transfer.unsqueeze(0)
    _check_cancelled(cancel_token)
    propagated_padded = torch.fft.ifft2(
        propagated_spectrum,
        dim=(-2, -1),
    )
    _check_cancelled(cancel_token)

    output_data = propagated_padded[crop_slices]
    if tuple(output_data.shape) != tuple(field.data.shape):
        raise RuntimeError(
            "Internal ASM crop did not restore canonical field shape: "
            f"expected {tuple(field.data.shape)}, got {tuple(output_data.shape)}."
        )
    output_field = field.with_data(
        output_data,
        grid=replace(field.grid,
                     x_center=field.grid.x_center + shift_x,
                     y_center=field.grid.y_center + shift_y),
        medium_index=medium_index,
        z_m=field.z_m + spec.distance_m,
    )
    power_after_full = _integrated_intensity_float(
        propagated_padded,
        output_field,
    )
    power_after = _integrated_intensity_float(output_data, output_field)
    crop_retained_power_fraction = (
        power_after / power_after_full if power_after_full > 0.0 else None
    )
    relative_full_power_change = (
        (power_after_full - power_before) / power_before
        if power_before > 0.0
        else None
    )
    elapsed = time.perf_counter() - started

    propagating_fraction = float(
        propagating_mask.to(dtype=real_dtype).mean().detach().item()
    )
    retained_fraction = float(
        transfer_mask.to(dtype=real_dtype).mean().detach().item()
    )
    diagnostics: Mapping[str, Any] = {
        "window_shift_m": (shift_x, shift_y),
        "window_sampling_note": "Periodic FFT sampling; shift does not prevent wrap-around. Centred sampling estimates do not certify shifted windows.",
        "identity": False,
        "input_shape": field.grid.shape,
        "computational_shape": computational_shape,
        "output_shape": output_field.grid.shape,
        "padding": padding,
        "dtype": str(target_dtype),
        "component_count": field.component_count,
        "vacuum_wavelength_m": field.wavelength_m,
        "medium_index_real": medium_index.real,
        "medium_index_imag": medium_index.imag,
        "k0_per_m": k0,
        "evanescent_policy": spec.evanescent_policy.value,
        "propagating_bin_fraction": propagating_fraction,
        "retained_bin_fraction": retained_fraction,
        "removed_input_spectral_energy_fraction": removed_spectral_fraction,
        "power_input": power_before,
        "power_output_full_computational": power_after_full,
        "power_output_cropped": power_after,
        "crop_retained_power_fraction": crop_retained_power_fraction,
        "relative_full_power_change": relative_full_power_change,
        "blas_cutoff_x_per_m": cutoff_x_per_m,
        "blas_cutoff_y_per_m": cutoff_y_per_m,
        "blas_retained_frequency_fraction": (
            retained_fraction
            if spec.method is PropagationMethod.BLAS
            else None
        ),
        "blas_removed_spectral_energy_fraction": (
            removed_spectral_fraction
            if spec.method is PropagationMethod.BLAS
            else None
        ),
        "bandlimit": (
            {
                "cutoff_x_per_m": cutoff_x_per_m,
                "cutoff_y_per_m": cutoff_y_per_m,
                "theoretical_cutoff_x_per_m": theoretical_cutoff_x,
                "theoretical_cutoff_y_per_m": theoretical_cutoff_y,
                "retained_frequency_fraction": retained_fraction,
                "removed_spectral_energy_fraction": removed_spectral_fraction,
                "support": "coupled_matsushima",
            }
            if spec.method is PropagationMethod.BLAS
            else None
        ),
        "blas_reference": (
            "Matsushima-Shimobaba 2009, DOI 10.1364/OE.17.019662"
            if spec.method is PropagationMethod.BLAS
            else None
        ),
        "sqrt_branch": "Re(kz)>=0; if Re(kz)=0 then Im(kz)>=0",
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


def _resolved_complex_dtype(
    field: Field2D,
    spec: PropagationSpec,
) -> torch.dtype:
    if spec.precision is PrecisionPolicy.COMPLEX64:
        return torch.complex64
    if spec.precision is PrecisionPolicy.COMPLEX128:
        return torch.complex128
    return field.data.dtype


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
    crop = (
        slice(None),
        slice(top, top + input_ny),
        slice(left, left + input_nx),
    )
    # Ordering follows torch pad: left, right, top, bottom.
    return padded, crop, (left, right, top, bottom)


def _radiation_branch(
    kz: torch.Tensor,
    *,
    k0: float,
    medium_index: complex,
) -> torch.Tensor:
    """Select the outgoing square-root branch without changing |kz|."""

    real_dtype = kz.real.dtype
    tolerance = (
        torch.finfo(real_dtype).eps
        * max(1.0, abs(k0 * medium_index))
        * 16.0
    )
    negative_real = kz.real < -tolerance
    on_imaginary_axis = torch.abs(kz.real) <= tolerance
    negative_imaginary = kz.imag < 0.0
    flip = negative_real | (on_imaginary_axis & negative_imaginary)
    return torch.where(flip, -kz, kz)


def _matsushima_cutoffs(
    *,
    wavelength_medium_m: float,
    distance_m: float,
    width_m: float,
    height_m: float,
) -> tuple[float, float]:
    absolute_distance = abs(distance_m)
    cutoff_x = 1.0 / (
        wavelength_medium_m
        * math.sqrt(1.0 + (2.0 * absolute_distance / width_m) ** 2)
    )
    cutoff_y = 1.0 / (
        wavelength_medium_m
        * math.sqrt(1.0 + (2.0 * absolute_distance / height_m) ** 2)
    )
    return cutoff_x, cutoff_y


def _removed_spectral_energy_fraction(
    spectrum: torch.Tensor,
    transfer_mask: torch.Tensor,
) -> float:
    energy = torch.abs(spectrum.detach()) ** 2
    total = energy.sum()
    if float(total.item()) == 0.0:
        return 0.0
    retained = (energy * transfer_mask.unsqueeze(0)).sum()
    fraction = 1.0 - float((retained / total).item())
    # Round-off can put the result a few ulps outside the probability range.
    return min(1.0, max(0.0, fraction))


def _integrated_intensity_float(
    data: torch.Tensor,
    field: Field2D,
) -> float:
    intensity = torch.sum(torch.abs(data.detach()) ** 2)
    return float((intensity * field.grid.dx * field.grid.dy).item())


def _check_cancelled(cancel_token: CancellationToken | None) -> None:
    if cancel_token is not None:
        cancel_token.throw_if_cancelled()


__all__ = [
    "AngularSpectrumPropagator",
    "propagate_angular_spectrum",
]
