"""Formula-derived tests for the canonical Rayleigh--Sommerfeld backends.

The numerical reference in this module is deliberately independent of the
production implementation and the legacy stateful class.  For the
``exp(-i omega t)`` convention it evaluates the first Rayleigh--Sommerfeld
solution directly:

    h_z(r) = z / (i lambda_m r**2) * exp(i k r)
             * (1 - 1 / (i k r)),  z > 0

and applies the input-cell area ``dx * dy`` as the quadrature weight.
Negative distance is rejected because a regular outgoing spatial kernel is
not the exact inverse of evanescent propagation.
"""

import math

import pytest
import torch

from optiprop.core import Field2D, Grid2D, ValidationError
from optiprop.propagation import (
    CancellationToken,
    PaddingSpec,
    PrecisionPolicy,
    PropagationCancelled,
    PropagationMethod,
    PropagationSpec,
    RayleighSommerfeldPropagator,
    assess_sampling,
    propagate_rayleigh_sommerfeld,
)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.complex64:
        return 2.5e-4, 2.5e-5
    return 2.0e-11, 2.0e-11


def _field(
    *,
    nx: int = 5,
    ny: int = 4,
    dx: float = 0.85e-6,
    dy: float = 1.25e-6,
    wavelength_m: float = 633e-9,
    medium_index: complex = 1.0,
    dtype: torch.dtype = torch.complex128,
    components: tuple[str, ...] = ("scalar",),
    device: str = "cpu",
    requires_grad: bool = False,
) -> Field2D:
    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    real_dtype = _real_dtype(dtype)
    generator = torch.Generator(device="cpu").manual_seed(1729)
    real = torch.randn(
        (len(components), ny, nx),
        dtype=real_dtype,
        generator=generator,
    )
    imag = torch.randn(
        (len(components), ny, nx),
        dtype=real_dtype,
        generator=generator,
    )
    data = torch.complex(real, imag).to(device=device)
    data.requires_grad_(requires_grad)
    return Field2D(
        data=data,
        grid=grid,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
        components=components,
        z_m=3.5e-6,
    )


def _spec(
    method: PropagationMethod,
    distance_m: float,
    *,
    padding: PaddingSpec | None = None,
    output_grid: Grid2D | None = None,
    precision: PrecisionPolicy = PrecisionPolicy.INHERIT,
    chunk_size: int | None = None,
) -> PropagationSpec:
    options = {} if chunk_size is None else {"chunk_size": chunk_size}
    return PropagationSpec(
        method=method,
        distance_m=distance_m,
        padding=PaddingSpec.auto() if padding is None else padding,
        output_grid=output_grid,
        precision=precision,
        options=options,
    )


def _direct_rs1_reference(
    field: Field2D,
    *,
    distance_m: float,
    output_grid: Grid2D,
    medium_index: complex | None = None,
) -> torch.Tensor:
    """Evaluate exact RS-I quadrature without using production helpers."""

    if distance_m == 0.0:
        if output_grid != field.grid:
            raise ValueError("The z=0 reference only defines same-grid identity.")
        return field.data.clone()

    dtype = field.dtype
    real_dtype = _real_dtype(dtype)
    device = field.device
    x_in, y_in = field.grid.meshgrid(dtype=real_dtype, device=device)
    x_out, y_out = output_grid.meshgrid(dtype=real_dtype, device=device)
    delta_x = x_out.reshape(-1, 1) - x_in.reshape(1, -1)
    delta_y = y_out.reshape(-1, 1) - y_in.reshape(1, -1)
    z = torch.as_tensor(distance_m, dtype=real_dtype, device=device)
    direction = 1.0 if distance_m > 0.0 else -1.0
    radius = torch.sqrt(delta_x.square() + delta_y.square() + z.square())

    index = field.medium_index if medium_index is None else complex(medium_index)
    wavelength_medium = field.wavelength_m / index
    wave_number = 2.0 * math.pi / wavelength_medium
    wave_number_tensor = torch.as_tensor(
        wave_number,
        dtype=dtype,
        device=device,
    )
    wavelength_tensor = torch.as_tensor(
        wavelength_medium,
        dtype=dtype,
        device=device,
    )
    radius_complex = radius.to(dtype)
    direction_complex = torch.as_tensor(direction, dtype=dtype, device=device)
    kernel = (
        z.to(dtype)
        / (1j * wavelength_tensor * radius_complex.square())
        * torch.exp(1j * direction_complex * wave_number_tensor * radius_complex)
        * (
            1.0
            - 1.0
            / (
                1j
                * direction_complex
                * wave_number_tensor
                * radius_complex
            )
        )
    )
    flattened = field.data.reshape(field.component_count, -1)
    result = flattened @ kernel.transpose(0, 1)
    result = result * (field.grid.dx * field.grid.dy)
    return result.reshape(
        field.component_count,
        output_grid.ny,
        output_grid.nx,
    )


@pytest.mark.parametrize(
    "method",
    [PropagationMethod.RS_FFT, PropagationMethod.RS_DIRECT],
)
def test_zero_distance_is_exact_identity(method: PropagationMethod) -> None:
    field = _field(nx=6, ny=5, components=("Ex", "Ey"))
    before = field.data.clone()

    result = propagate_rayleigh_sommerfeld(field, _spec(method, 0.0))

    assert result.field is not field
    assert result.field.grid == field.grid
    assert result.field.components == ("Ex", "Ey")
    assert result.field.z_m == pytest.approx(field.z_m)
    torch.testing.assert_close(result.field.data, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(field.data, before, rtol=0.0, atol=0.0)
    assert result.diagnostics["identity"] is True


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_rs_direct_matches_independent_reference_on_shifted_output(
    dtype: torch.dtype,
) -> None:
    field = _field(
        nx=4,
        ny=3,
        dx=0.72e-6,
        dy=1.31e-6,
        dtype=dtype,
        components=("Ex", "Ey"),
    )
    output_grid = Grid2D(
        nx=3,
        ny=5,
        dx=0.93e-6,
        dy=0.68e-6,
        x_center=1.17e-6,
        y_center=-0.44e-6,
    )
    distance_m = 12.7e-6
    before = field.data.clone()

    result = propagate_rayleigh_sommerfeld(
        field,
        _spec(
            PropagationMethod.RS_DIRECT,
            distance_m,
            output_grid=output_grid,
            chunk_size=3,
        ),
    )
    expected = _direct_rs1_reference(
        field,
        distance_m=distance_m,
        output_grid=output_grid,
    )

    rtol, atol = _tolerances(dtype)
    assert result.field.grid == output_grid
    assert result.field.data.shape == (2, 5, 3)
    assert result.field.dtype == dtype
    assert result.field.z_m == pytest.approx(field.z_m + distance_m)
    torch.testing.assert_close(result.field.data, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(field.data, before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("nx", "ny"),
    [(5, 4), (4, 5), (5, 5), (4, 4)],
)
def test_rs_fft_matches_direct_for_rectangular_odd_even_grids(
    nx: int,
    ny: int,
) -> None:
    field = _field(nx=nx, ny=ny, dx=0.79e-6, dy=1.43e-6)
    distance_m = 18.0e-6

    fft_result = propagate_rayleigh_sommerfeld(
        field,
        _spec(PropagationMethod.RS_FFT, distance_m),
    )
    direct_result = propagate_rayleigh_sommerfeld(
        field,
        _spec(
            PropagationMethod.RS_DIRECT,
            distance_m,
            output_grid=field.grid,
            chunk_size=4,
        ),
    )

    assert fft_result.field.grid == field.grid
    torch.testing.assert_close(
        fft_result.field.data,
        direct_result.field.data,
        rtol=3.0e-11,
        atol=3.0e-11,
    )


def test_rs_fft_is_linear_convolution_without_opposite_edge_wrap() -> None:
    grid = Grid2D(nx=9, ny=6, dx=0.83e-6, dy=1.27e-6)
    data = torch.zeros((1, grid.ny, grid.nx), dtype=torch.complex128)
    data[0, 1, 0] = 1.0 + 0.25j
    field = Field2D(data=data, grid=grid, wavelength_m=532e-9)
    distance_m = 7.0e-6

    result = propagate_rayleigh_sommerfeld(
        field,
        _spec(PropagationMethod.RS_FFT, distance_m),
    )
    expected = _direct_rs1_reference(
        field,
        distance_m=distance_m,
        output_grid=grid,
    )

    torch.testing.assert_close(
        result.field.data,
        expected,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    assert result.diagnostics["linear_convolution"] is True


def test_rs_fft_rejects_padding_that_cannot_embed_linear_convolution() -> None:
    field = _field(nx=6, ny=4)
    spec = _spec(
        PropagationMethod.RS_FFT,
        7.0e-6,
        padding=PaddingSpec.none(),
    )
    propagator = RayleighSommerfeldPropagator()

    report = propagator.validate(field, spec)
    assert not report.is_valid
    assert "rayleigh_sommerfeld.fft_padding_too_small" in {
        issue.code for issue in report.errors
    }
    with pytest.raises(ValidationError) as error:
        propagator.propagate(field, spec)
    assert "rayleigh_sommerfeld.fft_padding_too_small" in {
        issue.code for issue in error.value.report.errors
    }


def test_positive_distance_locks_global_phase_and_rs1_correction() -> None:
    """A one-cell quadrature point exposes the complete complex RS-I kernel."""

    grid = Grid2D(nx=1, ny=1, dx=0.31e-6, dy=0.47e-6)
    field = Field2D(
        data=torch.ones((1, 1, 1), dtype=torch.complex128),
        grid=grid,
        wavelength_m=632.8e-9,
        medium_index=1.37,
    )
    distance = 4.3e-6

    positive = propagate_rayleigh_sommerfeld(
        field,
        _spec(
            PropagationMethod.RS_DIRECT,
            distance,
            output_grid=grid,
            chunk_size=1,
        ),
    ).field.data
    reference = _direct_rs1_reference(
        field,
        distance_m=distance,
        output_grid=grid,
    )

    torch.testing.assert_close(positive, reference, rtol=2e-12, atol=2e-12)


def test_passive_complex_index_matches_exact_direct_reference() -> None:
    field = _field(
        nx=4,
        ny=3,
        medium_index=1.42 + 0.015j,
        dtype=torch.complex128,
    )
    distance_m = 7.0e-6
    expected = _direct_rs1_reference(
        field,
        distance_m=distance_m,
        output_grid=field.grid,
    )

    direct = propagate_rayleigh_sommerfeld(
        field,
        _spec(
            PropagationMethod.RS_DIRECT,
            distance_m,
            output_grid=field.grid,
            chunk_size=2,
        ),
    )
    fft = propagate_rayleigh_sommerfeld(
        field,
        _spec(PropagationMethod.RS_FFT, distance_m),
    )

    torch.testing.assert_close(
        direct.field.data,
        expected,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        fft.field.data,
        expected,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


@pytest.mark.parametrize(
    "method",
    [PropagationMethod.RS_FFT, PropagationMethod.RS_DIRECT],
)
def test_negative_distance_is_rejected_as_unsupported(
    method: PropagationMethod,
) -> None:
    """RS spatial convolution is not an exact evanescent-wave inverse."""

    field = _field(nx=3, ny=2)
    spec = _spec(method, -8.0e-6, chunk_size=2)
    propagator = RayleighSommerfeldPropagator()

    report = propagator.validate(field, spec)
    assert not report.is_valid
    assert "rs.negative_distance_unsupported" in {
        issue.code for issue in report.errors
    }
    with pytest.raises(ValidationError) as error:
        propagator.propagate(field, spec)
    assert "rs.negative_distance_unsupported" in {
        issue.code for issue in error.value.report.errors
    }


@pytest.mark.parametrize("components", [("scalar",), ("Ex", "Ey")])
def test_both_backends_preserve_components(
    components: tuple[str, ...],
) -> None:
    field = _field(nx=4, ny=3, components=components)
    for method in (PropagationMethod.RS_FFT, PropagationMethod.RS_DIRECT):
        result = propagate_rayleigh_sommerfeld(
            field,
            _spec(method, 9.0e-6, chunk_size=2),
        )
        assert result.field.components == components
        assert result.field.data.shape == (
            len(components),
            field.grid.ny,
            field.grid.nx,
        )


@pytest.mark.parametrize(
    ("input_dtype", "precision", "expected_dtype"),
    [
        (
            torch.complex128,
            PrecisionPolicy.COMPLEX64,
            torch.complex64,
        ),
        (
            torch.complex64,
            PrecisionPolicy.COMPLEX128,
            torch.complex128,
        ),
    ],
)
def test_precision_override_is_honored(
    input_dtype: torch.dtype,
    precision: PrecisionPolicy,
    expected_dtype: torch.dtype,
) -> None:
    field = _field(nx=3, ny=2, dtype=input_dtype)
    result = propagate_rayleigh_sommerfeld(
        field,
        _spec(
            PropagationMethod.RS_DIRECT,
            8.0e-6,
            precision=precision,
            chunk_size=2,
        ),
    )
    assert result.field.dtype == expected_dtype
    assert field.dtype == input_dtype


@pytest.mark.parametrize(
    "method",
    [PropagationMethod.RS_FFT, PropagationMethod.RS_DIRECT],
)
def test_input_is_not_mutated_and_autograd_reaches_input(
    method: PropagationMethod,
) -> None:
    field = _field(
        nx=4,
        ny=3,
        dtype=torch.complex128,
        requires_grad=True,
    )
    before = field.data.detach().clone()

    result = propagate_rayleigh_sommerfeld(
        field,
        _spec(method, 11.0e-6, chunk_size=2),
    )
    loss = result.field.intensity().sum()
    loss.backward()

    torch.testing.assert_close(field.data.detach(), before, rtol=0.0, atol=0.0)
    assert field.data.grad is not None
    assert torch.isfinite(field.data.grad).all()
    assert torch.count_nonzero(field.data.grad).item() > 0


@pytest.mark.parametrize(
    "wrong_method",
    [PropagationMethod.ASM, PropagationMethod.FRESNEL_TF],
)
def test_backend_rejects_non_rs_methods(
    wrong_method: PropagationMethod,
) -> None:
    field = _field(nx=3, ny=2)
    spec = _spec(wrong_method, 10.0e-6)
    propagator = RayleighSommerfeldPropagator()

    report = propagator.validate(field, spec)
    assert not report.is_valid
    assert "propagation.unsupported_method" in {
        issue.code for issue in report.errors
    }
    with pytest.raises(ValidationError) as error:
        propagator.propagate(field, spec)
    assert "propagation.unsupported_method" in {
        issue.code for issue in error.value.report.errors
    }


def test_direct_result_exposes_exact_workload_preflight() -> None:
    field = _field(nx=5, ny=4, components=("Ex", "Ey"))
    output_grid = Grid2D(nx=3, ny=7, dx=0.7e-6, dy=0.9e-6)
    spec = _spec(
        PropagationMethod.RS_DIRECT,
        13e-6,
        output_grid=output_grid,
        chunk_size=4,
    )

    preflight = assess_sampling(field, spec)
    result = propagate_rayleigh_sommerfeld(field, spec)

    expected_operations = 2 * (5 * 4) * (3 * 7)
    assert preflight.estimated_operations == expected_operations
    assert result.sampling is not None
    assert result.sampling.estimated_operations == expected_operations
    issue = next(
        item
        for item in result.validation.issues
        if item.code == "sampling.rs_direct_workload"
    )
    assert issue.details["estimated_operations"] == expected_operations


def test_direct_workload_thresholds_produce_structured_warning_and_error() -> None:
    field = _field(nx=5, ny=4, components=("Ex", "Ey"))
    output_grid = Grid2D(nx=4, ny=3, dx=0.7e-6, dy=0.9e-6)
    spec = _spec(
        PropagationMethod.RS_DIRECT,
        13e-6,
        output_grid=output_grid,
    )

    warning = assess_sampling(
        field,
        spec,
        rs_direct_warning_operations=100,
        rs_direct_error_operations=1000,
    )
    error = assess_sampling(
        field,
        spec,
        rs_direct_warning_operations=10,
        rs_direct_error_operations=100,
    )

    warning_issue = next(
        item
        for item in warning.validation.issues
        if item.code == "sampling.rs_direct_workload"
    )
    error_issue = next(
        item
        for item in error.validation.issues
        if item.code == "sampling.rs_direct_workload"
    )
    assert warning.is_valid
    assert warning_issue.severity.value == "warning"
    assert not error.is_valid
    assert error_issue.severity.value == "error"


@pytest.mark.parametrize("bad_chunk_size", [0, -1, 1.5, True])
def test_direct_rejects_invalid_chunk_size(bad_chunk_size: object) -> None:
    field = _field(nx=3, ny=2)
    spec = PropagationSpec(
        method=PropagationMethod.RS_DIRECT,
        distance_m=8.0e-6,
        padding=PaddingSpec.auto(),
        options={"chunk_size": bad_chunk_size},
    )
    propagator = RayleighSommerfeldPropagator()

    report = propagator.validate(field, spec)
    assert not report.is_valid
    assert "rayleigh_sommerfeld.invalid_chunk_size" in {
        issue.code for issue in report.errors
    }
    with pytest.raises(ValidationError):
        propagator.propagate(field, spec)


def test_direct_automatic_chunk_respects_memory_budget() -> None:
    field = _field(nx=20, ny=10, dtype=torch.complex128)
    spec = PropagationSpec(
        method=PropagationMethod.RS_DIRECT,
        distance_m=8.0e-6,
        padding=PaddingSpec.auto(),
        options={"direct_memory_budget_bytes": 200 * 12 * 16 * 3},
    )

    result = propagate_rayleigh_sommerfeld(field, spec)

    assert result.diagnostics["chunk_selection"] == "automatic_memory_budget"
    assert result.diagnostics["chunk_size"] == 3
    assert result.diagnostics["estimated_peak_interaction_bytes"] <= (
        result.diagnostics["direct_memory_budget_bytes"]
    )


def test_rs_validation_warns_when_kernel_phase_is_undersampled() -> None:
    field = _field(
        nx=9,
        ny=7,
        dx=1.5e-6,
        dy=1.8e-6,
        wavelength_m=500e-9,
    )
    spec = _spec(
        PropagationMethod.RS_FFT,
        0.5e-6,
        padding=PaddingSpec.auto(),
    )

    report = RayleighSommerfeldPropagator().validate(field, spec)

    assert report.is_valid
    warning_codes = {issue.code for issue in report.warnings}
    assert "rayleigh_sommerfeld.kernel_sampling_x" in warning_codes
    assert "rayleigh_sommerfeld.kernel_sampling_y" in warning_codes


@pytest.mark.parametrize("bad_budget", [0, -1, 1.5, True])
def test_direct_rejects_invalid_memory_budget(bad_budget: object) -> None:
    field = _field(nx=3, ny=2)
    spec = PropagationSpec(
        method=PropagationMethod.RS_DIRECT,
        distance_m=8.0e-6,
        padding=PaddingSpec.auto(),
        options={"direct_memory_budget_bytes": bad_budget},
    )

    report = RayleighSommerfeldPropagator().validate(field, spec)

    assert not report.is_valid
    assert "rayleigh_sommerfeld.invalid_memory_budget" in {
        issue.code for issue in report.errors
    }
    with pytest.raises(ValidationError):
        propagate_rayleigh_sommerfeld(field, spec)


@pytest.mark.parametrize(
    "method",
    [PropagationMethod.RS_FFT, PropagationMethod.RS_DIRECT],
)
def test_pre_cancelled_request_stops_before_work(
    method: PropagationMethod,
) -> None:
    token = CancellationToken()
    token.cancel("test cancellation")

    with pytest.raises(PropagationCancelled, match="test cancellation"):
        propagate_rayleigh_sommerfeld(
            _field(nx=4, ny=3),
            _spec(method, 10e-6, chunk_size=2),
            cancel_token=token,
        )


class _CancelAfterChecks(CancellationToken):
    def __init__(self, cancel_at: int) -> None:
        super().__init__()
        self.cancel_at = cancel_at
        self.check_count = 0

    def throw_if_cancelled(self) -> None:
        self.check_count += 1
        if self.check_count == self.cancel_at:
            self.cancel("cancelled between direct chunks")
        super().throw_if_cancelled()


def test_direct_checks_cancellation_between_chunks() -> None:
    token = _CancelAfterChecks(cancel_at=4)
    field = _field(nx=8, ny=7)
    output_grid = Grid2D(nx=7, ny=6, dx=0.7e-6, dy=1.1e-6)

    with pytest.raises(
        PropagationCancelled,
        match="cancelled between direct chunks",
    ):
        propagate_rayleigh_sommerfeld(
            field,
            _spec(
                PropagationMethod.RS_DIRECT,
                12e-6,
                output_grid=output_grid,
                chunk_size=2,
            ),
            cancel_token=token,
        )
    assert token.check_count == 4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "method",
    [PropagationMethod.RS_FFT, PropagationMethod.RS_DIRECT],
)
def test_cuda_matches_cpu(method: PropagationMethod) -> None:
    cpu_field = _field(nx=5, ny=4, dtype=torch.complex128)
    gpu_field = cpu_field.to(device="cuda")
    spec = _spec(method, 14.0e-6, chunk_size=3)

    cpu_result = propagate_rayleigh_sommerfeld(cpu_field, spec)
    gpu_result = propagate_rayleigh_sommerfeld(gpu_field, spec)

    assert gpu_result.field.device.type == "cuda"
    torch.testing.assert_close(
        gpu_result.field.data.cpu(),
        cpu_result.field.data,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
