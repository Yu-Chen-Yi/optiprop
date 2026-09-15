"""Reference and behavioral tests for the canonical Fresnel backends.

The tests derive expected values from the paraxial transfer function and the
single-FFT Fresnel diffraction integral.  They intentionally avoid comparing
against the legacy implementation.
"""

import math

import pytest
import torch

from optiprop.core import Field2D, Grid2D, ValidationError
from optiprop.propagation import (
    AngularSpectrumPropagator,
    CancellationToken,
    FresnelPropagator,
    PaddingSpec,
    PropagationCancelled,
    PropagationMethod,
    PropagationSpec,
)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.complex64:
        return 7.0e-5, 7.0e-5
    return 3.0e-11, 3.0e-11


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
) -> tuple[Field2D, float, float]:
    """Return a periodic FFT mode and its physical spatial frequencies."""

    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    x_grid, y_grid = grid.meshgrid(
        dtype=_real_dtype(dtype),
        device=device,
    )
    frequency_x = mode_x / (nx * dx)
    frequency_y = mode_y / (ny * dy)
    phase = 2.0 * math.pi * (
        frequency_x * x_grid + frequency_y * y_grid
    )
    data = torch.exp(1j * phase).to(dtype).unsqueeze(0)
    return (
        Field2D(
            data=data,
            grid=grid,
            wavelength_m=wavelength_m,
            medium_index=medium_index,
        ),
        frequency_x,
        frequency_y,
    )


def _spec(
    method: PropagationMethod,
    distance_m: float,
    *,
    padding: PaddingSpec | None = None,
    output_grid: Grid2D | None = None,
) -> PropagationSpec:
    return PropagationSpec(
        method=method,
        distance_m=distance_m,
        padding=PaddingSpec.none() if padding is None else padding,
        output_grid=output_grid,
    )


def _propagate(field: Field2D, spec: PropagationSpec):
    return FresnelPropagator().propagate(field, spec)


@pytest.mark.parametrize(
    "method",
    [PropagationMethod.FRESNEL_TF, PropagationMethod.FRESNEL_SCALED],
)
def test_zero_distance_is_exact_identity(method: PropagationMethod) -> None:
    field, _, _ = _plane_wave(
        nx=18,
        ny=13,
        dx=0.9e-6,
        dy=1.2e-6,
        mode_x=2,
        mode_y=-1,
        wavelength_m=633e-9,
    )
    before = field.data.clone()

    result = _propagate(field, _spec(method, 0.0))

    assert result.field is not field
    assert result.field.grid == field.grid
    assert result.field.components == field.components
    assert result.field.z_m == pytest.approx(field.z_m)
    torch.testing.assert_close(result.field.data, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(field.data, before, rtol=0.0, atol=0.0)
    assert result.diagnostics["identity"] is True


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_transfer_function_rectangular_plane_wave_matches_paraxial_phase(
    dtype: torch.dtype,
) -> None:
    """Lock [ny,nx], independent dx/dy, phase sign, k0*n and precision."""

    wavelength_m = 633e-9
    medium_index = 1.33
    distance_m = 73e-6
    field, frequency_x, frequency_y = _plane_wave(
        nx=96,
        ny=71,
        dx=0.80e-6,
        dy=1.30e-6,
        mode_x=5,
        mode_y=-3,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
        dtype=dtype,
    )
    before = field.data.clone()

    result = _propagate(
        field,
        _spec(PropagationMethod.FRESNEL_TF, distance_m),
    )

    k = 2.0 * math.pi * medium_index / wavelength_m
    kx = 2.0 * math.pi * frequency_x
    ky = 2.0 * math.pi * frequency_y
    paraxial_kz = k - (kx * kx + ky * ky) / (2.0 * k)
    transfer = torch.exp(
        torch.as_tensor(
            1j * paraxial_kz * distance_m,
            dtype=dtype,
            device=field.device,
        )
    )
    rtol, atol = _tolerances(dtype)
    assert result.field.data.shape == (1, 71, 96)
    assert result.field.dtype == dtype
    assert result.field.grid == field.grid
    assert result.field.z_m == pytest.approx(field.z_m + distance_m)
    torch.testing.assert_close(
        result.field.data,
        before * transfer,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(field.data, before, rtol=0.0, atol=0.0)


def test_transfer_function_complex_index_has_expected_absorption() -> None:
    wavelength_m = 1.0e-6
    medium_index = 1.4 + 0.012j
    distance_m = 9.0e-6
    grid = Grid2D(nx=20, ny=15, dx=0.8e-6, dy=1.1e-6)
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
        _spec(PropagationMethod.FRESNEL_TF, distance_m),
    )

    expected = data * torch.exp(
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
        expected,
        rtol=3.0e-11,
        atol=3.0e-11,
    )


@pytest.mark.parametrize(
    "method",
    [PropagationMethod.FRESNEL_TF, PropagationMethod.FRESNEL_SCALED],
)
def test_ex_ey_batch_matches_two_independent_scalar_runs(
    method: PropagationMethod,
) -> None:
    grid = Grid2D(nx=19, ny=15, dx=1.2e-6, dy=1.8e-6)
    generator = torch.Generator().manual_seed(20260730)
    ex_data = torch.complex(
        torch.randn((1, grid.ny, grid.nx), generator=generator),
        torch.randn((1, grid.ny, grid.nx), generator=generator),
    ).to(torch.complex128)
    ey_data = torch.complex(
        torch.randn((1, grid.ny, grid.nx), generator=generator),
        torch.randn((1, grid.ny, grid.nx), generator=generator),
    ).to(torch.complex128)
    ex = Field2D(data=ex_data, grid=grid, wavelength_m=780e-9)
    ey = Field2D(data=ey_data, grid=grid, wavelength_m=780e-9)
    combined_data = torch.cat((ex_data, ey_data), dim=0)
    combined = Field2D(
        data=combined_data,
        grid=grid,
        wavelength_m=780e-9,
        components=("Ex", "Ey"),
    )
    before = combined.data.clone()
    spec = _spec(method, 0.8e-3)

    batch = _propagate(combined, spec)
    scalar_ex = _propagate(ex, spec)
    scalar_ey = _propagate(ey, spec)

    assert batch.field.components == ("Ex", "Ey")
    torch.testing.assert_close(
        batch.field.data,
        torch.cat((scalar_ex.field.data, scalar_ey.field.data), dim=0),
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    torch.testing.assert_close(combined.data, before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "method",
    [PropagationMethod.FRESNEL_TF, PropagationMethod.FRESNEL_SCALED],
)
def test_fresnel_preserves_autograd_and_does_not_mutate_input(
    method: PropagationMethod,
) -> None:
    grid = Grid2D(nx=17, ny=13, dx=1.4e-6, dy=2.1e-6)
    real = torch.randn((1, grid.ny, grid.nx), dtype=torch.float64)
    imag = torch.randn((1, grid.ny, grid.nx), dtype=torch.float64)
    data = torch.complex(real, imag).requires_grad_()
    before = data.detach().clone()
    field = Field2D(data=data, grid=grid, wavelength_m=633e-9)

    result = _propagate(field, _spec(method, 1.1e-3))
    loss = result.field.intensity().mean()
    loss.backward()

    assert result.field.data.requires_grad
    assert data.grad is not None
    assert bool(torch.isfinite(data.grad.real).all())
    assert bool(torch.isfinite(data.grad.imag).all())
    assert float(torch.linalg.vector_norm(data.grad)) > 0.0
    torch.testing.assert_close(field.data.detach(), before, rtol=0.0, atol=0.0)


def test_transfer_function_agrees_with_asm_in_paraxial_regime() -> None:
    field, _, _ = _plane_wave(
        nx=128,
        ny=95,
        dx=4.0e-6,
        dy=6.0e-6,
        mode_x=3,
        mode_y=-2,
        wavelength_m=633e-9,
        dtype=torch.complex128,
    )
    distance_m = 1.0e-3
    fresnel_spec = _spec(PropagationMethod.FRESNEL_TF, distance_m)
    asm_spec = PropagationSpec(
        method=PropagationMethod.ASM,
        distance_m=distance_m,
        padding=PaddingSpec.none(),
    )

    fresnel = _propagate(field, fresnel_spec)
    exact = AngularSpectrumPropagator().propagate(field, asm_spec)

    torch.testing.assert_close(
        fresnel.field.data,
        exact.field.data,
        # The leading neglected term in sqrt(k^2-kt^2) produces a
        # deterministic 4.34e-7 rad phase difference for this mode.
        rtol=6.0e-7,
        atol=6.0e-7,
    )


def test_scaled_fresnel_natural_grid_and_global_prefactor() -> None:
    """A centred discrete impulse has the analytic spherical chirp output."""

    wavelength_m = 633e-9
    medium_index = 1.20
    wavelength_medium_m = wavelength_m / medium_index
    distance_m = 20e-3
    grid = Grid2D(nx=17, ny=13, dx=2.0e-6, dy=3.0e-6)
    data = torch.zeros((1, grid.ny, grid.nx), dtype=torch.complex128)
    data[0, grid.ny // 2, grid.nx // 2] = 1.0
    field = Field2D(
        data=data,
        grid=grid,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
    )

    result = _propagate(
        field,
        _spec(PropagationMethod.FRESNEL_SCALED, distance_m),
    )

    expected_dx = wavelength_medium_m * distance_m / (grid.nx * grid.dx)
    expected_dy = wavelength_medium_m * distance_m / (grid.ny * grid.dy)
    assert result.field.grid.nx == grid.nx
    assert result.field.grid.ny == grid.ny
    assert result.field.grid.dx == pytest.approx(expected_dx, rel=1.0e-14)
    assert result.field.grid.dy == pytest.approx(expected_dy, rel=1.0e-14)
    assert result.field.grid.x_center == pytest.approx(grid.x_center)
    assert result.field.grid.y_center == pytest.approx(grid.y_center)

    output_x, output_y = result.field.grid.meshgrid(dtype=torch.float64)
    k = 2.0 * math.pi * medium_index / wavelength_m
    global_prefactor = torch.exp(
        torch.as_tensor(
            1j * k * distance_m,
            dtype=torch.complex128,
        )
    ) / (1j * wavelength_medium_m * distance_m)
    output_chirp = torch.exp(
        1j * k * (output_x.square() + output_y.square()) / (2.0 * distance_m)
    )
    expected = (
        grid.dx
        * grid.dy
        * global_prefactor
        * output_chirp.to(torch.complex128)
    ).unsqueeze(0)
    torch.testing.assert_close(
        result.field.data,
        expected,
        rtol=4.0e-11,
        atol=2.0e-17,
    )


@pytest.mark.parametrize("distance_m", [1.7e-3, -1.7e-3])
def test_scaled_fresnel_matches_direct_integral_with_shifted_even_grid(
    distance_m: float,
) -> None:
    wavelength_m = 780e-9
    medium_index = 1.25
    wavelength_medium = wavelength_m / medium_index
    input_grid = Grid2D(
        nx=6,
        ny=4,
        dx=1.1e-6,
        dy=1.7e-6,
        x_center=0.4e-6,
        y_center=-0.7e-6,
    )
    output_grid = Grid2D(
        nx=input_grid.nx,
        ny=input_grid.ny,
        dx=(
            wavelength_medium
            * abs(distance_m)
            / (input_grid.nx * input_grid.dx)
        ),
        dy=(
            wavelength_medium
            * abs(distance_m)
            / (input_grid.ny * input_grid.dy)
        ),
        x_center=13.0e-6,
        y_center=-9.0e-6,
    )
    generator = torch.Generator().manual_seed(404)
    data = torch.complex(
        torch.randn(
            (1, input_grid.ny, input_grid.nx),
            dtype=torch.float64,
            generator=generator,
        ),
        torch.randn(
            (1, input_grid.ny, input_grid.nx),
            dtype=torch.float64,
            generator=generator,
        ),
    )
    field = Field2D(
        data=data,
        grid=input_grid,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
    )

    result = _propagate(
        field,
        _spec(
            PropagationMethod.FRESNEL_SCALED,
            distance_m,
            output_grid=output_grid,
        ),
    )

    x1, y1 = input_grid.meshgrid(dtype=torch.float64)
    x2, y2 = output_grid.meshgrid(dtype=torch.float64)
    k = 2.0 * math.pi * medium_index / wavelength_m
    pre_chirped = data[0] * torch.exp(
        1j * k * (x1.square() + y1.square()) / (2.0 * distance_m)
    )
    kernel = torch.exp(
        -1j
        * k
        / distance_m
        * (
            x2.reshape(-1, 1) * x1.reshape(1, -1)
            + y2.reshape(-1, 1) * y1.reshape(1, -1)
        )
    )
    integral = (
        kernel @ pre_chirped.reshape(-1, 1)
    ).reshape(output_grid.shape) * input_grid.dx * input_grid.dy
    expected = (
        torch.exp(
            torch.as_tensor(
                1j * k * distance_m,
                dtype=torch.complex128,
            )
        )
        / (1j * wavelength_medium * distance_m)
        * torch.exp(
            1j
            * k
            * (x2.square() + y2.square())
            / (2.0 * distance_m)
        )
        * integral
    ).unsqueeze(0)

    torch.testing.assert_close(
        result.field.data,
        expected,
        rtol=4.0e-11,
        atol=4.0e-11,
    )


def test_transfer_function_round_trip_and_scaled_power_conservation() -> None:
    grid = Grid2D(nx=18, ny=13, dx=1.3e-6, dy=1.9e-6)
    generator = torch.Generator().manual_seed(405)
    data = torch.complex(
        torch.randn(
            (1, grid.ny, grid.nx),
            dtype=torch.float64,
            generator=generator,
        ),
        torch.randn(
            (1, grid.ny, grid.nx),
            dtype=torch.float64,
            generator=generator,
        ),
    )
    field = Field2D(data=data, grid=grid, wavelength_m=633e-9)
    distance_m = 0.8e-3

    forward = _propagate(
        field,
        _spec(PropagationMethod.FRESNEL_TF, distance_m),
    )
    restored = _propagate(
        forward.field,
        _spec(PropagationMethod.FRESNEL_TF, -distance_m),
    )
    scaled = _propagate(
        field,
        _spec(PropagationMethod.FRESNEL_SCALED, distance_m),
    )

    torch.testing.assert_close(
        restored.field.data,
        field.data,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    assert scaled.power_after / scaled.power_before == pytest.approx(
        1.0,
        rel=3.0e-12,
    )


def test_scaled_fresnel_accepts_exact_natural_output_grid() -> None:
    wavelength_m = 532e-9
    medium_index = 1.4
    distance_m = 4e-3
    grid = Grid2D(nx=21, ny=16, dx=1.1e-6, dy=1.7e-6)
    natural_grid = Grid2D(
        nx=grid.nx,
        ny=grid.ny,
        dx=(wavelength_m / medium_index) * distance_m / (grid.nx * grid.dx),
        dy=(wavelength_m / medium_index) * distance_m / (grid.ny * grid.dy),
        x_center=grid.x_center,
        y_center=grid.y_center,
    )
    field = Field2D(
        data=torch.ones((1, grid.ny, grid.nx), dtype=torch.complex128),
        grid=grid,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
    )

    result = _propagate(
        field,
        _spec(
            PropagationMethod.FRESNEL_SCALED,
            distance_m,
            output_grid=natural_grid,
        ),
    )

    assert result.field.grid == natural_grid
    assert result.field.data.shape == (1, natural_grid.ny, natural_grid.nx)


def test_scaled_fresnel_rejects_non_natural_output_grid() -> None:
    grid = Grid2D(nx=21, ny=16, dx=1.1e-6, dy=1.7e-6)
    field = Field2D(
        data=torch.ones((1, grid.ny, grid.nx), dtype=torch.complex128),
        grid=grid,
        wavelength_m=532e-9,
    )
    incompatible = Grid2D(
        nx=grid.nx,
        ny=grid.ny,
        dx=7.0e-6,
        dy=9.0e-6,
    )
    spec = _spec(
        PropagationMethod.FRESNEL_SCALED,
        4e-3,
        output_grid=incompatible,
    )
    propagator = FresnelPropagator()

    report = propagator.validate(field, spec)

    assert not report.is_valid
    assert any(issue.parameter_path == "output_grid" for issue in report.errors)
    with pytest.raises(ValidationError):
        propagator.propagate(field, spec)


def test_transfer_function_padding_crops_back_to_input_grid() -> None:
    grid = Grid2D(nx=31, ny=24, dx=1.2e-6, dy=1.9e-6)
    data = torch.zeros((1, grid.ny, grid.nx), dtype=torch.complex128)
    data[0, grid.ny // 2, grid.nx // 2] = 1.0
    field = Field2D(data=data, grid=grid, wavelength_m=633e-9)

    result = _propagate(
        field,
        _spec(
            PropagationMethod.FRESNEL_TF,
            0.5e-3,
            padding=PaddingSpec.by_factor(2.0),
        ),
    )

    assert result.sampling is not None
    assert result.sampling.computational_grid.shape == (48, 62)
    assert result.field.grid == grid
    assert result.field.data.shape == (1, grid.ny, grid.nx)
    assert result.diagnostics["computational_shape"] == (48, 62)


def test_scaled_fresnel_padding_returns_natural_computational_grid() -> None:
    wavelength_m = 633e-9
    distance_m = 0.5e-3
    grid = Grid2D(nx=31, ny=24, dx=1.2e-6, dy=1.9e-6)
    data = torch.zeros((1, grid.ny, grid.nx), dtype=torch.complex128)
    data[0, grid.ny // 2, grid.nx // 2] = 1.0
    field = Field2D(data=data, grid=grid, wavelength_m=wavelength_m)

    result = _propagate(
        field,
        _spec(
            PropagationMethod.FRESNEL_SCALED,
            distance_m,
            padding=PaddingSpec.by_factor(2.0),
        ),
    )

    assert result.sampling is not None
    assert result.sampling.computational_grid.shape == (48, 62)
    assert result.field.data.shape == (1, 48, 62)
    assert result.field.grid.shape == (48, 62)
    assert result.field.grid.dx == pytest.approx(
        wavelength_m * distance_m / (62 * grid.dx)
    )
    assert result.field.grid.dy == pytest.approx(
        wavelength_m * distance_m / (48 * grid.dy)
    )


@pytest.mark.parametrize(
    "method",
    [
        PropagationMethod.ASM,
        PropagationMethod.BLAS,
        PropagationMethod.RS_FFT,
        PropagationMethod.RS_DIRECT,
    ],
)
def test_unsupported_method_is_structured_validation_error(
    method: PropagationMethod,
) -> None:
    field, _, _ = _plane_wave(
        nx=12,
        ny=9,
        dx=1.0e-6,
        dy=1.2e-6,
        mode_x=1,
        mode_y=1,
        wavelength_m=633e-9,
    )
    spec = _spec(method, 10e-6)
    propagator = FresnelPropagator()

    report = propagator.validate(field, spec)

    assert not report.is_valid
    assert "propagation.unsupported_method" in {
        issue.code for issue in report.errors
    }
    with pytest.raises(ValidationError):
        propagator.propagate(field, spec)


def test_pre_cancelled_request_raises_without_mutating_inputs() -> None:
    field, _, _ = _plane_wave(
        nx=20,
        ny=15,
        dx=0.8e-6,
        dy=1.1e-6,
        mode_x=2,
        mode_y=-1,
        wavelength_m=633e-9,
    )
    before = field.data.clone()
    spec = _spec(PropagationMethod.FRESNEL_TF, 50e-6)
    token = CancellationToken()
    token.cancel("test cancellation")

    with pytest.raises(PropagationCancelled, match="test cancellation"):
        FresnelPropagator().propagate(field, spec, token)

    torch.testing.assert_close(field.data, before, rtol=0.0, atol=0.0)
    assert spec.distance_m == pytest.approx(50e-6)
    assert spec.method is PropagationMethod.FRESNEL_TF


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_transfer_function_plane_wave_matches_paraxial_phase() -> None:
    wavelength_m = 633e-9
    distance_m = 80e-6
    field, frequency_x, frequency_y = _plane_wave(
        nx=48,
        ny=35,
        dx=0.8e-6,
        dy=1.1e-6,
        mode_x=3,
        mode_y=-2,
        wavelength_m=wavelength_m,
        dtype=torch.complex64,
        device="cuda",
    )

    result = _propagate(
        field,
        _spec(PropagationMethod.FRESNEL_TF, distance_m),
    )

    k = 2.0 * math.pi / wavelength_m
    kx = 2.0 * math.pi * frequency_x
    ky = 2.0 * math.pi * frequency_y
    paraxial_kz = k - (kx * kx + ky * ky) / (2.0 * k)
    expected = field.data * torch.exp(
        torch.as_tensor(
            1j * paraxial_kz * distance_m,
            dtype=torch.complex64,
            device="cuda",
        )
    )
    assert result.field.device.type == "cuda"
    torch.testing.assert_close(
        result.field.data,
        expected,
        rtol=2.0e-4,
        atol=2.0e-4,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_scaled_fresnel_matches_cpu() -> None:
    grid = Grid2D(nx=17, ny=14, dx=1.2e-6, dy=1.8e-6)
    generator = torch.Generator().manual_seed(406)
    cpu_data = torch.complex(
        torch.randn(
            (1, grid.ny, grid.nx),
            generator=generator,
        ),
        torch.randn(
            (1, grid.ny, grid.nx),
            generator=generator,
        ),
    ).to(torch.complex64)
    cpu_field = Field2D(
        data=cpu_data,
        grid=grid,
        wavelength_m=633e-9,
    )
    gpu_field = cpu_field.to("cuda")
    spec = _spec(PropagationMethod.FRESNEL_SCALED, 1.2e-3)

    cpu_result = _propagate(cpu_field, spec)
    gpu_result = _propagate(gpu_field, spec)

    assert gpu_result.field.device.type == "cuda"
    assert gpu_result.field.grid == cpu_result.field.grid
    torch.testing.assert_close(
        gpu_result.field.data.cpu(),
        cpu_result.field.data,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
