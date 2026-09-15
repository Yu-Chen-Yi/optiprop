"""Reference and behavioral tests for the OptiProp 2.0 ASM/BLAS backend."""

import math

import pytest
import torch

from optiprop.core import Field2D, Grid2D, ValidationError
from optiprop.propagation import (
    AngularSpectrumPropagator,
    CancellationToken,
    EvanescentPolicy,
    PaddingSpec,
    PropagationCancelled,
    PropagationMethod,
    PropagationSpec,
)
from optiprop.propagation.asm import _pad_to_shape


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.complex64:
        return 3.0e-5, 3.0e-5
    return 2.0e-11, 2.0e-11


def _plane_wave(
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    mode_x: int,
    mode_y: int,
    wavelength_m: float,
    medium_index: float = 1.0,
    dtype: torch.dtype = torch.complex128,
    device: str = "cpu",
) -> tuple[Field2D, float]:
    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    real_dtype = _real_dtype(dtype)
    x_grid, y_grid = grid.meshgrid(dtype=real_dtype, device=device)
    fx = mode_x / (nx * dx)
    fy = mode_y / (ny * dy)
    phase = 2.0 * math.pi * (fx * x_grid + fy * y_grid)
    data = torch.exp(1j * phase).to(dtype).unsqueeze(0)
    k = 2.0 * math.pi * medium_index / wavelength_m
    kx = 2.0 * math.pi * fx
    ky = 2.0 * math.pi * fy
    kz = math.sqrt(k * k - kx * kx - ky * ky)
    return (
        Field2D(
            data=data,
            grid=grid,
            wavelength_m=wavelength_m,
            medium_index=medium_index,
        ),
        kz,
    )


def _spec(
    method: PropagationMethod,
    distance_m: float,
    *,
    policy: EvanescentPolicy = EvanescentPolicy.DISCARD,
    padding: PaddingSpec | None = None,
) -> PropagationSpec:
    return PropagationSpec(
        method=method,
        distance_m=distance_m,
        evanescent_policy=policy,
        padding=PaddingSpec.none() if padding is None else padding,
    )


def _propagate(field: Field2D, spec: PropagationSpec):
    return AngularSpectrumPropagator().propagate(field, spec)


@pytest.mark.parametrize("method", [PropagationMethod.ASM, PropagationMethod.BLAS])
def test_zero_distance_is_exact_identity(method: PropagationMethod) -> None:
    field, _ = _plane_wave(
        nx=18,
        ny=13,
        dx=0.9e-6,
        dy=1.2e-6,
        mode_x=2,
        mode_y=-1,
        wavelength_m=633e-9,
        dtype=torch.complex128,
    )
    before = field.data.clone()
    spec = _spec(method, 0.0)

    result = _propagate(field, spec)

    assert result.field is not field
    assert result.field.grid == field.grid
    assert result.field.components == field.components
    assert result.field.z_m == pytest.approx(field.z_m)
    torch.testing.assert_close(result.field.data, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(field.data, before)
    assert "sampling.zero_distance" in {
        issue.code for issue in result.validation.issues
    }


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_rectangular_discrete_plane_wave_matches_analytic_eigenvalue(
    dtype: torch.dtype,
) -> None:
    """This simultaneously locks [ny,nx], dx/dy, phase sign and precision."""
    field, kz = _plane_wave(
        nx=192,
        ny=127,
        dx=0.80e-6,
        dy=1.30e-6,
        mode_x=7,
        mode_y=-5,
        wavelength_m=633e-9,
        medium_index=1.33,
        dtype=dtype,
    )
    distance_m = 37.0e-6
    before = field.data.clone()
    spec = _spec(PropagationMethod.ASM, distance_m)

    result = _propagate(field, spec)

    expected = before * torch.exp(
        torch.as_tensor(
            1j * kz * distance_m,
            dtype=dtype,
            device=before.device,
        )
    )
    rtol, atol = _tolerances(dtype)
    assert result.field.data.shape == (1, 127, 192)
    assert result.field.data.dtype == dtype
    assert result.field.grid.dx == pytest.approx(0.80e-6)
    assert result.field.grid.dy == pytest.approx(1.30e-6)
    assert result.field.z_m == pytest.approx(field.z_m + distance_m)
    torch.testing.assert_close(result.field.data, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(field.data, before)


def test_ex_ey_batch_matches_two_independent_scalar_runs() -> None:
    ex, _ = _plane_wave(
        nx=32,
        ny=23,
        dx=0.7e-6,
        dy=1.1e-6,
        mode_x=3,
        mode_y=-2,
        wavelength_m=780e-9,
        dtype=torch.complex128,
    )
    ey, _ = _plane_wave(
        nx=32,
        ny=23,
        dx=0.7e-6,
        dy=1.1e-6,
        mode_x=-4,
        mode_y=1,
        wavelength_m=780e-9,
        dtype=torch.complex128,
    )
    combined_data = torch.cat((ex.data, 0.35j * ey.data), dim=0)
    combined_before = combined_data.clone()
    combined = Field2D(
        data=combined_data,
        grid=ex.grid,
        wavelength_m=ex.wavelength_m,
        components=("Ex", "Ey"),
    )
    scalar_ey = ey.with_data(0.35j * ey.data)
    spec = _spec(PropagationMethod.ASM, 55e-6)

    batch_result = _propagate(combined, spec)
    ex_result = _propagate(ex, spec)
    ey_result = _propagate(scalar_ey, spec)

    expected = torch.cat((ex_result.field.data, ey_result.field.data), dim=0)
    assert batch_result.field.components == ("Ex", "Ey")
    torch.testing.assert_close(
        batch_result.field.data,
        expected,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(combined.data, combined_before)


def test_discard_policy_forward_backward_round_trip_for_bandlimited_field() -> None:
    grid = Grid2D(nx=40, ny=31, dx=0.8e-6, dy=1.0e-6)
    spectrum = torch.zeros((grid.ny, grid.nx), dtype=torch.complex64)
    spectrum[0, 0] = 1.0
    spectrum[2, 3] = 0.4 + 0.2j
    spectrum[-3, 5] = -0.3 + 0.1j
    spectrum[4, -6] = 0.15 - 0.25j
    data = torch.fft.ifft2(spectrum).unsqueeze(0)
    field = Field2D(data=data, grid=grid, wavelength_m=700e-9)
    before = field.data.clone()
    forward_spec = _spec(PropagationMethod.ASM, 90e-6)
    backward_spec = _spec(PropagationMethod.ASM, -90e-6)

    forward = _propagate(field, forward_spec)
    restored = _propagate(forward.field, backward_spec)

    assert restored.validation.is_valid
    torch.testing.assert_close(
        restored.field.data,
        before,
        rtol=4.0e-5,
        atol=2.0e-7,
    )
    torch.testing.assert_close(field.data, before)


def test_decay_policy_forward_is_finite_and_attenuates_evanescent_mode() -> None:
    wavelength_m = 1.0e-6
    grid = Grid2D(
        nx=32,
        ny=24,
        dx=wavelength_m / 4,
        dy=wavelength_m / 5,
    )
    checker = (1 - 2 * (torch.arange(grid.nx) % 2)).to(torch.float64)
    data = checker.repeat(grid.ny, 1).to(torch.complex128).unsqueeze(0)
    field = Field2D(data=data, grid=grid, wavelength_m=wavelength_m)
    before = field.data.clone()
    spec = _spec(
        PropagationMethod.ASM,
        2.0 * wavelength_m,
        policy=EvanescentPolicy.DECAY,
    )

    result = _propagate(field, spec)

    assert bool(torch.isfinite(result.field.data.real).all())
    assert bool(torch.isfinite(result.field.data.imag).all())
    assert torch.linalg.vector_norm(result.field.data) < torch.linalg.vector_norm(
        before
    )
    torch.testing.assert_close(field.data, before)


def test_complex_index_plane_wave_has_expected_absorption() -> None:
    wavelength_m = 1.0e-6
    medium_index = 1.45 + 0.015j
    distance_m = 8.0e-6
    grid = Grid2D(nx=20, ny=17, dx=0.7e-6, dy=0.9e-6)
    data = torch.ones(
        (1, grid.ny, grid.nx),
        dtype=torch.complex128,
    )
    field = Field2D(
        data=data,
        grid=grid,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
    )

    result = _propagate(
        field,
        _spec(PropagationMethod.ASM, distance_m),
    )

    expected_transfer = torch.exp(
        torch.as_tensor(
            1j
            * (2.0 * math.pi / wavelength_m)
            * medium_index
            * distance_m,
            dtype=torch.complex128,
        )
    )
    torch.testing.assert_close(
        result.field.data,
        data * expected_transfer,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    expected_power_ratio = math.exp(
        -2.0
        * (2.0 * math.pi / wavelength_m)
        * medium_index.imag
        * distance_m
    )
    assert result.power_after / result.power_before == pytest.approx(
        expected_power_ratio,
        rel=2.0e-12,
    )
    assert "sampling.complex_index_approximation" in {
        issue.code for issue in result.warnings
    }


def test_asm_preserves_autograd_graph() -> None:
    grid = Grid2D(nx=18, ny=15, dx=0.8e-6, dy=1.1e-6)
    real = torch.randn((1, grid.ny, grid.nx), dtype=torch.float64)
    imag = torch.randn((1, grid.ny, grid.nx), dtype=torch.float64)
    data = torch.complex(real, imag).requires_grad_()
    field = Field2D(data=data, grid=grid, wavelength_m=633e-9)

    result = _propagate(
        field,
        _spec(PropagationMethod.ASM, 12.0e-6),
    )
    loss = result.field.intensity().mean()
    loss.backward()

    assert result.field.data.requires_grad
    assert data.grad is not None
    assert bool(torch.isfinite(data.grad.real).all())
    assert bool(torch.isfinite(data.grad.imag).all())
    assert float(torch.linalg.vector_norm(data.grad)) > 0.0


def test_decay_policy_negative_distance_is_stable_regularized_decay() -> None:
    wavelength_m = 1.0e-6
    grid = Grid2D(
        nx=32,
        ny=24,
        dx=wavelength_m / 4,
        dy=wavelength_m / 5,
    )
    checker = (1 - 2 * (torch.arange(grid.nx) % 2)).to(torch.float64)
    data = checker.repeat(grid.ny, 1).to(torch.complex128).unsqueeze(0)
    field = Field2D(data=data, grid=grid, wavelength_m=wavelength_m)
    spec = _spec(
        PropagationMethod.ASM,
        -2.0 * wavelength_m,
        policy=EvanescentPolicy.DECAY,
    )

    result = _propagate(field, spec)

    assert result.validation.is_valid
    assert torch.linalg.vector_norm(result.field.data) < torch.linalg.vector_norm(
        field.data
    )


@pytest.mark.parametrize("method", [PropagationMethod.ASM, PropagationMethod.BLAS])
def test_padding_is_cropped_back_to_original_shape_and_grid(
    method: PropagationMethod,
) -> None:
    field, _ = _plane_wave(
        nx=31,
        ny=25,
        dx=0.65e-6,
        dy=0.90e-6,
        mode_x=2,
        mode_y=-1,
        wavelength_m=633e-9,
        dtype=torch.complex64,
    )
    before = field.data.clone()
    spec = _spec(method, 40e-6, padding=PaddingSpec.by_factor(2.0))

    result = _propagate(field, spec)

    assert result.sampling is not None
    assert result.sampling.computational_grid.shape == (50, 62)
    assert result.field.data.shape == before.shape == (1, 25, 31)
    assert result.field.grid == field.grid
    assert result.spec is spec
    torch.testing.assert_close(field.data, before)
    assert spec.padding == PaddingSpec.by_factor(2.0)


@pytest.mark.parametrize(
    ("input_shape", "target_shape"),
    [
        ((5, 7), (9, 11)),   # odd -> odd
        ((5, 7), (8, 10)),   # odd -> even
        ((6, 8), (9, 11)),   # even -> odd
        ((6, 8), (10, 12)),  # even -> even
    ],
)
def test_padding_crop_is_bitwise_exact_for_all_parity_transitions(
    input_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> None:
    ny, nx = input_shape
    data = torch.complex(
        torch.arange(ny * nx, dtype=torch.float64).reshape(1, ny, nx),
        torch.zeros((1, ny, nx), dtype=torch.float64),
    )

    padded, crop, _ = _pad_to_shape(data, target_shape)

    assert padded.shape[-2:] == target_shape
    torch.testing.assert_close(padded[crop], data, rtol=0.0, atol=0.0)


def test_blas_reports_cutoffs_and_removes_out_of_band_energy() -> None:
    wavelength_m = 900e-9
    grid = Grid2D(
        nx=32,
        ny=24,
        dx=wavelength_m / 3,
        dy=wavelength_m / 4,
    )
    data = torch.zeros((1, grid.ny, grid.nx), dtype=torch.complex128)
    data[0, grid.ny // 2, grid.nx // 2] = 1.0
    field = Field2D(data=data, grid=grid, wavelength_m=wavelength_m)
    spec = _spec(PropagationMethod.BLAS, 80e-6)

    result = _propagate(field, spec)

    bandlimit = result.diagnostics["bandlimit"]
    assert 0.0 < bandlimit["cutoff_x_per_m"] <= 1.0 / (2 * grid.dx)
    assert 0.0 < bandlimit["cutoff_y_per_m"] <= 1.0 / (2 * grid.dy)
    assert 0.0 < bandlimit["retained_frequency_fraction"] < 1.0
    assert 0.0 < bandlimit["removed_spectral_energy_fraction"] < 1.0
    assert float(result.field.power()) < float(field.power())


def test_blas_uses_coupled_boundary_not_rectangular_axial_cutoffs() -> None:
    """A corner bin can pass both 1-D limits while violating 2-D BLAS bounds."""
    wavelength_m = 1.0e-6
    grid = Grid2D(
        nx=64,
        ny=64,
        dx=wavelength_m / 4,
        dy=wavelength_m / 4,
    )
    distance_m = 4.0e-6
    frequency_x = torch.fft.fftfreq(
        grid.nx, d=grid.dx, dtype=torch.float64
    )
    frequency_y = torch.fft.fftfreq(
        grid.ny, d=grid.dy, dtype=torch.float64
    )
    fy, fx = torch.meshgrid(frequency_y, frequency_x, indexing="ij")
    width_m = grid.nx * grid.dx
    height_m = grid.ny * grid.dy
    inverse_lambda_squared = 1.0 / wavelength_m**2
    x_scale = 1.0 + (2.0 * distance_m / width_m) ** 2
    y_scale = 1.0 + (2.0 * distance_m / height_m) ** 2
    cutoff_x = 1.0 / (wavelength_m * math.sqrt(x_scale))
    cutoff_y = 1.0 / (wavelength_m * math.sqrt(y_scale))

    inside_rectangular_cutoffs = (
        (fx.abs() <= cutoff_x)
        & (fy.abs() <= cutoff_y)
        & (fx.square() + fy.square() <= inverse_lambda_squared)
    )
    inside_coupled_x = (
        x_scale * fx.square() + fy.square() <= inverse_lambda_squared
    )
    inside_coupled_y = (
        fx.square() + y_scale * fy.square() <= inverse_lambda_squared
    )
    corner_candidates = inside_rectangular_cutoffs & ~(
        inside_coupled_x & inside_coupled_y
    )
    assert bool(corner_candidates.any()), "test grid must contain a corner bin"
    candidate_y, candidate_x = torch.nonzero(
        corner_candidates, as_tuple=False
    )[0]

    spectrum = torch.zeros((grid.ny, grid.nx), dtype=torch.complex128)
    spectrum[candidate_y, candidate_x] = 1.0
    field = Field2D(
        data=torch.fft.ifft2(spectrum).unsqueeze(0),
        grid=grid,
        wavelength_m=wavelength_m,
    )

    asm = _propagate(
        field,
        _spec(PropagationMethod.ASM, distance_m),
    )
    blas = _propagate(
        field,
        _spec(PropagationMethod.BLAS, distance_m),
    )

    assert float(asm.field.power()) == pytest.approx(
        float(field.power()), rel=2.0e-12
    )
    assert float(blas.field.power()) < float(field.power()) * 1.0e-20
    assert blas.diagnostics["bandlimit"][
        "removed_spectral_energy_fraction"
    ] == pytest.approx(1.0, abs=1.0e-14)


def test_asm_and_blas_agree_for_plane_wave_inside_safe_band() -> None:
    field, _ = _plane_wave(
        nx=64,
        ny=48,
        dx=0.8e-6,
        dy=1.0e-6,
        mode_x=1,
        mode_y=-1,
        wavelength_m=700e-9,
        dtype=torch.complex128,
    )
    distance_m = 20e-6

    asm = _propagate(field, _spec(PropagationMethod.ASM, distance_m))
    blas = _propagate(field, _spec(PropagationMethod.BLAS, distance_m))

    bandlimit = blas.diagnostics["bandlimit"]
    assert bandlimit["removed_spectral_energy_fraction"] == pytest.approx(
        0.0, abs=1.0e-14
    )
    torch.testing.assert_close(
        blas.field.data,
        asm.field.data,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_power_diagnostics_distinguish_full_grid_from_crop() -> None:
    field, _ = _plane_wave(
        nx=48,
        ny=35,
        dx=0.8e-6,
        dy=1.1e-6,
        mode_x=2,
        mode_y=-1,
        wavelength_m=633e-9,
        dtype=torch.complex128,
    )

    periodic = _propagate(
        field,
        _spec(PropagationMethod.ASM, 15e-6),
    )
    padded = _propagate(
        field,
        _spec(
            PropagationMethod.ASM,
            15e-6,
            padding=PaddingSpec.by_factor(2.0),
        ),
    )

    assert periodic.diagnostics["relative_full_power_change"] == pytest.approx(
        0.0,
        abs=2.0e-12,
    )
    assert periodic.diagnostics["crop_retained_power_fraction"] == pytest.approx(
        1.0,
        abs=2.0e-12,
    )
    assert padded.diagnostics["power_output_full_computational"] >= (
        padded.diagnostics["power_output_cropped"]
    )
    assert 0.0 <= padded.diagnostics["crop_retained_power_fraction"] <= 1.0


@pytest.mark.parametrize(
    "method",
    [
        PropagationMethod.FRESNEL_TF,
        PropagationMethod.FRESNEL_SCALED,
        PropagationMethod.RS_FFT,
        PropagationMethod.RS_DIRECT,
    ],
)
def test_unsupported_method_is_structured_validation_error(
    method: PropagationMethod,
) -> None:
    field, _ = _plane_wave(
        nx=12,
        ny=9,
        dx=1.0e-6,
        dy=1.2e-6,
        mode_x=1,
        mode_y=1,
        wavelength_m=633e-9,
    )
    spec = _spec(method, 10e-6)
    propagator = AngularSpectrumPropagator()

    report = propagator.validate(field, spec)

    assert not report.is_valid
    assert "propagation.unsupported_method" in {
        issue.code for issue in report.errors
    }
    with pytest.raises(ValidationError):
        propagator.propagate(field, spec)


def test_pre_cancelled_request_raises_without_mutating_inputs() -> None:
    field, _ = _plane_wave(
        nx=20,
        ny=15,
        dx=0.8e-6,
        dy=1.1e-6,
        mode_x=2,
        mode_y=-1,
        wavelength_m=633e-9,
    )
    before = field.data.clone()
    spec = _spec(PropagationMethod.ASM, 50e-6)
    token = CancellationToken()
    token.cancel("test cancellation")

    with pytest.raises(PropagationCancelled, match="test cancellation"):
        AngularSpectrumPropagator().propagate(field, spec, token)

    torch.testing.assert_close(field.data, before)
    assert spec.distance_m == pytest.approx(50e-6)
    assert spec.method is PropagationMethod.ASM


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_plane_wave_matches_analytic_phase() -> None:
    field, kz = _plane_wave(
        nx=48,
        ny=35,
        dx=0.8e-6,
        dy=1.1e-6,
        mode_x=3,
        mode_y=-2,
        wavelength_m=633e-9,
        dtype=torch.complex64,
        device="cuda",
    )
    distance_m = 30e-6
    spec = _spec(PropagationMethod.ASM, distance_m)

    result = _propagate(field, spec)
    expected = field.data * torch.exp(
        torch.as_tensor(
            1j * kz * distance_m,
            dtype=torch.complex64,
            device="cuda",
        )
    )

    assert result.field.data.device.type == "cuda"
    torch.testing.assert_close(
        result.field.data,
        expected,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
