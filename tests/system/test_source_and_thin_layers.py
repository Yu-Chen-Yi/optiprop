"""Independent physics/contract tests for source and thin-element layers."""

from __future__ import annotations

import json
import math

import pytest
import torch

from optiprop import Field2D, Grid2D, ValidationError
from optiprop.system import (
    ApertureLayer,
    ApertureShape,
    ApertureSpec,
    ComplexMaskLayer,
    IdealLensLayer,
    LensPhaseModel,
    MaskSamplingMode,
    OpticalLayer,
    OpticalSystem,
    SourceKind,
    IncidentSource,
)


def _grid(*, nx: int = 9, ny: int = 7) -> Grid2D:
    return Grid2D(nx=nx, ny=ny, dx=1e-6, dy=1.5e-6)


def _field(
    *,
    grid: Grid2D | None = None,
    components: tuple[str, ...] = ("scalar",),
    dtype: torch.dtype = torch.complex128,
    device: str = "cpu",
    requires_grad: bool = False,
    medium_index: complex = 1.0,
) -> Field2D:
    grid = _grid() if grid is None else grid
    data = torch.ones(
        (len(components), grid.ny, grid.nx),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return Field2D(
        data=data,
        grid=grid,
        wavelength_m=500e-9,
        medium_index=medium_index,
        components=components,
    )


def test_plane_wave_has_exact_amplitude_phase_and_metadata() -> None:
    grid = _grid()
    layer = IncidentSource.plane_wave(
        grid,
        500e-9,
        amplitudes=(2.5,),
        phase_offset_rad=0.37,
        medium_index=1.4,
        name="uniform",
    )

    field = layer.create(dtype=torch.complex128)
    expected = torch.full_like(
        field.data,
        2.5 * complex(math.cos(0.37), math.sin(0.37)),
    )

    # A source creates the initial field and is intentionally not a mid-chain
    # field-in/field-out OpticalLayer.
    assert not isinstance(layer, OpticalLayer)
    assert layer.kind is SourceKind.PLANE_WAVE
    assert field.grid == grid
    assert field.wavelength_m == pytest.approx(500e-9)
    assert field.medium_index == 1.4 + 0j
    assert field.components == ("scalar",)
    torch.testing.assert_close(field.data, expected, rtol=0, atol=1e-15)


def test_tilted_plane_wave_locks_positive_spatial_phase_convention() -> None:
    grid = _grid(nx=11, ny=9)
    angle_x = 0.08
    angle_y = -0.04
    wavelength = 633e-9
    refractive_index = 1.25
    layer = IncidentSource.tilted_plane_wave(
        grid,
        wavelength,
        angle_x_rad=angle_x,
        angle_y_rad=angle_y,
        medium_index=refractive_index,
    )

    field = layer.create(dtype=torch.complex128)
    k = 2 * math.pi * refractive_index / wavelength
    expected_x_ratio = torch.tensor(
        complex(
            math.cos(k * math.sin(angle_x) * grid.dx),
            math.sin(k * math.sin(angle_x) * grid.dx),
        ),
        dtype=field.dtype,
    )
    expected_y_ratio = torch.tensor(
        complex(
            math.cos(k * math.sin(angle_y) * grid.dy),
            math.sin(k * math.sin(angle_y) * grid.dy),
        ),
        dtype=field.dtype,
    )

    center_y, center_x = grid.ny // 2, grid.nx // 2
    torch.testing.assert_close(
        field.data[0, center_y, center_x + 1]
        / field.data[0, center_y, center_x],
        expected_x_ratio,
    )
    torch.testing.assert_close(
        field.data[0, center_y + 1, center_x]
        / field.data[0, center_y, center_x],
        expected_y_ratio,
    )


def test_elliptical_gaussian_uses_one_over_e_amplitude_waists() -> None:
    grid = Grid2D(nx=9, ny=9, dx=1e-6, dy=1e-6)
    layer = IncidentSource.gaussian(
        grid,
        532e-9,
        waist_x_m=2e-6,
        waist_y_m=3e-6,
        amplitudes=(3.0,),
    )

    field = layer.create(dtype=torch.complex128)
    c = grid.nx // 2

    assert layer.kind is SourceKind.ELLIPTICAL_GAUSSIAN
    assert abs(field.data[0, c, c]) == pytest.approx(3.0)
    assert abs(field.data[0, c, c + 2]) == pytest.approx(3.0 / math.e)
    assert abs(field.data[0, c + 3, c]) == pytest.approx(3.0 / math.e)


def test_gaussian_source_can_also_carry_tilt_phase() -> None:
    grid = _grid(nx=11, ny=9)
    angle_x = 0.05
    wavelength = 633e-9
    layer = IncidentSource.gaussian(
        grid,
        wavelength,
        waist_x_m=4e-6,
        angle_x_rad=angle_x,
    )

    field = layer.create(dtype=torch.complex128)
    c_y, c_x = grid.ny // 2, grid.nx // 2
    ratio = (
        field.data[0, c_y, c_x + 1] / field.data[0, c_y, c_x]
    )
    unit_ratio = ratio / abs(ratio)
    expected_phase = (
        2 * math.pi / wavelength * math.sin(angle_x) * grid.dx
    )

    torch.testing.assert_close(
        unit_ratio,
        torch.tensor(
            complex(math.cos(expected_phase), math.sin(expected_phase)),
            dtype=field.dtype,
        ),
    )


def test_jones_source_components_are_explicit_and_not_shape_inferred() -> None:
    layer = IncidentSource.plane_wave(
        _grid(),
        500e-9,
        components=("Ex", "Ey"),
        amplitudes=(1.0, 1.0j),
    )

    field = layer.create(dtype=torch.complex64)

    assert field.components == ("Ex", "Ey")
    torch.testing.assert_close(field.data[0], torch.ones_like(field.data[0]))
    torch.testing.assert_close(
        field.data[1],
        torch.full_like(field.data[1], 1j),
    )


def test_amplitude_phase_source_is_exact_and_snapshots_inputs() -> None:
    grid = _grid()
    amplitude = torch.linspace(
        0.1, 1.0, grid.nx * grid.ny, dtype=torch.float64
    ).reshape(grid.ny, grid.nx)
    phase = torch.linspace(
        -math.pi, math.pi, grid.nx * grid.ny, dtype=torch.float64
    ).reshape(grid.ny, grid.nx)
    layer = IncidentSource.from_amplitude_phase(
        amplitude,
        phase,
        grid,
        500e-9,
    )
    expected = amplitude * torch.exp(1j * phase)

    amplitude.fill_(9)
    phase.zero_()
    field = layer.create(dtype=torch.complex128)

    torch.testing.assert_close(field.data[0], expected)


def test_imported_source_create_returns_independent_copy() -> None:
    incoming = _field(requires_grad=True)
    expected_imported = incoming.data.detach().clone()
    source = IncidentSource.plane_wave(
        incoming.grid,
        incoming.wavelength_m,
        amplitudes=(2.0,),
    )
    imported = IncidentSource.from_field(incoming)
    with torch.no_grad():
        incoming.data.fill_(9)

    generated = source.create(dtype=torch.complex128)
    recreated = imported.create()

    torch.testing.assert_close(
        generated.data,
        torch.full_like(generated.data, 2.0),
    )
    assert recreated is not incoming
    assert recreated.data is not incoming.data
    torch.testing.assert_close(recreated.data, expected_imported)


def test_disabled_source_is_reported_and_cannot_create_field() -> None:
    source = IncidentSource.plane_wave(
        _grid(),
        500e-9,
        enabled=False,
    )

    report = source.validate()

    assert not report.is_valid
    assert report.errors[0].code == "source.disabled"
    with pytest.raises(ValidationError):
        source.create()


def test_active_medium_source_is_layer_addressed_and_cannot_create() -> None:
    source = IncidentSource.plane_wave(
        _grid(),
        500e-9,
        medium_index=1.0 - 0.01j,
    )

    report = source.validate()

    assert not report.is_valid
    issue = report.errors[0]
    assert issue.code == "source.active_medium"
    assert issue.layer_id == str(source.id)
    assert issue.parameter_path == "medium_index"
    with pytest.raises(ValidationError):
        source.create()


def test_imported_source_snapshot_preserves_autograd_to_original_tensor() -> None:
    field = _field(requires_grad=True)
    source = IncidentSource.from_field(field)

    output = source.create()
    output.intensity().sum().backward()

    assert field.data.grad is not None
    assert torch.isfinite(field.data.grad).all()


def test_ideal_lens_matches_analytic_phase_and_positive_focus_sign() -> None:
    field = _field()
    focal_length = 80e-6
    layer = IdealLensLayer(
        focal_length_m=focal_length,
        design_wavelength_m=field.wavelength_m,
        transmission_amplitude=0.8,
        phase_offset_rad=0.2,
        phase_model=LensPhaseModel.PARAXIAL,
    )

    output = layer.apply(field)
    x, y = field.grid.meshgrid(dtype=torch.float64)
    expected = 0.8 * torch.exp(
        1j
        * (
            0.2
            - (2 * math.pi / field.wavelength_m)
            * (x.square() + y.square())
            / (2 * focal_length)
        )
    )

    torch.testing.assert_close(output.data[0], expected)
    assert output.grid == field.grid
    assert output.z_m == field.z_m
    assert output.medium_index == field.medium_index


def test_default_lens_uses_exact_equal_path_phase() -> None:
    field = _field()
    focal_length = 12e-6
    layer = IdealLensLayer(
        focal_length_m=focal_length,
        design_wavelength_m=field.wavelength_m,
    )

    output = layer.apply(field)
    x, y = field.grid.meshgrid(dtype=torch.float64)
    optical_path_excess = torch.sqrt(
        x.square() + y.square() + focal_length**2
    ) - abs(focal_length)
    expected = torch.exp(
        -1j * (2 * math.pi / field.wavelength_m) * optical_path_excess
    )

    assert layer.phase_model is LensPhaseModel.EXACT_EQUAL_PATH
    torch.testing.assert_close(output.data[0], expected)


def test_negative_focal_length_reverses_exact_lens_phase() -> None:
    field = _field()
    positive = IdealLensLayer(focal_length_m=12e-6).apply(field)
    negative = IdealLensLayer(focal_length_m=-12e-6).apply(field)

    torch.testing.assert_close(negative.data, positive.data.conj())


def test_lens_rejects_active_amplitude_gain() -> None:
    with pytest.raises(ValueError, match="amplitude|between"):
        IdealLensLayer(
            focal_length_m=20e-6,
            transmission_amplitude=1.01,
        )


def test_circular_aperture_includes_boundary_pixel_centers() -> None:
    grid = Grid2D(nx=5, ny=5, dx=1e-6, dy=1e-6)
    field = _field(grid=grid)
    aperture = ApertureSpec(
        shape=ApertureShape.CIRCLE,
        size_x_m=2e-6,
        edge_inclusive=True,
    )

    output = ApertureLayer(aperture).apply(field)

    expected = torch.zeros_like(output.data)
    expected[0, 2, 2] = 1
    expected[0, 1, 2] = 1
    expected[0, 3, 2] = 1
    expected[0, 2, 1] = 1
    expected[0, 2, 3] = 1
    torch.testing.assert_close(output.data, expected)


def test_complex_mask_broadcasts_to_ex_ey_and_preserves_autograd() -> None:
    field = _field(
        components=("Ex", "Ey"),
        requires_grad=True,
    )
    x, y = field.grid.meshgrid(dtype=torch.float64)
    transmission = (0.5 + 0.1 * x / field.grid.dx) * torch.exp(
        1j * 0.2 * y / field.grid.dy
    )
    layer = ComplexMaskLayer(
        transmission=transmission,
        grid=field.grid,
    )
    before = field.data.detach().clone()

    output = layer.apply(field)
    output.intensity().sum().backward()

    torch.testing.assert_close(output.data, before * transmission[None])
    torch.testing.assert_close(field.data.detach(), before, rtol=0, atol=0)
    assert field.data.grad is not None
    assert torch.isfinite(field.data.grad).all()


def test_complex_mask_snapshots_transmission_for_stable_hash_and_output() -> None:
    field = _field()
    transmission = torch.full(
        field.grid.shape,
        0.25 + 0.5j,
        dtype=torch.complex128,
    )
    layer = ComplexMaskLayer(transmission=transmission, grid=field.grid)
    hash_before = layer.parameter_hash

    transmission.fill_(9 + 0j)
    output = layer.apply(field)

    assert layer.parameter_hash == hash_before
    torch.testing.assert_close(
        output.data,
        torch.full_like(output.data, 0.25 + 0.5j),
    )


def test_complex_mask_clone_preserves_gradient_to_caller_tensor() -> None:
    field = _field()
    transmission = torch.full(
        field.grid.shape,
        0.5 + 0.25j,
        dtype=torch.complex128,
        requires_grad=True,
    )
    layer = ComplexMaskLayer(transmission=transmission, grid=field.grid)

    output = layer.apply(field)
    output.intensity().sum().backward()

    assert transmission.grad is not None
    assert torch.isfinite(transmission.grad).all()


def test_nonfinite_mask_is_layer_addressed_and_blocks_application() -> None:
    field = _field()
    transmission = torch.ones(
        field.grid.shape,
        dtype=torch.complex128,
    )
    transmission[0, 0] = complex(float("nan"), 0)
    layer = ComplexMaskLayer(transmission=transmission, grid=field.grid)

    report = layer.validate(field)

    assert not report.is_valid
    issue = next(
        item
        for item in report.errors
        if item.code == "mask.nonfinite_transmission"
    )
    assert issue.layer_id == str(layer.id)
    assert issue.parameter_path == "transmission"
    with pytest.raises(ValidationError):
        layer.apply(field)


def test_gain_mask_warns_with_maximum_magnitude_but_remains_executable() -> None:
    field = _field()
    layer = ComplexMaskLayer(
        transmission=torch.full(
            field.grid.shape,
            1.2 + 0j,
            dtype=torch.complex128,
        ),
        grid=field.grid,
    )

    report = layer.validate(field)
    output = layer.apply(field)

    assert report.is_valid
    issue = next(
        item
        for item in report.warnings
        if item.code == "mask.gain_transmission"
    )
    assert issue.details["maximum_magnitude"] == pytest.approx(1.2)
    torch.testing.assert_close(output.data, field.data * 1.2)


def test_strict_mask_grid_mismatch_is_layer_addressed_and_blocks_apply() -> None:
    field = _field()
    mask_grid = Grid2D(nx=5, ny=5, dx=2e-6, dy=2e-6)
    layer = ComplexMaskLayer(
        transmission=torch.ones(mask_grid.shape, dtype=torch.complex128),
        grid=mask_grid,
        sampling_mode=MaskSamplingMode.STRICT,
    )

    report = layer.validate(field)

    assert not report.is_valid
    assert all(issue.layer_id == str(layer.id) for issue in report.errors)
    assert report.errors[0].parameter_path in {
        "grid",
        "sampling_mode",
        "transmission",
    }
    with pytest.raises(ValidationError):
        layer.apply(field)


def test_explicit_complex_resampling_returns_field_grid_without_mutation() -> None:
    field = _field()
    mask_grid = Grid2D(nx=5, ny=5, dx=2e-6, dy=3e-6)
    mask = torch.full(mask_grid.shape, 0.5 + 0.25j, dtype=torch.complex128)
    layer = ComplexMaskLayer(
        transmission=mask,
        grid=mask_grid,
        sampling_mode=MaskSamplingMode.RESAMPLE_COMPLEX,
    )
    before = field.data.clone()

    output = layer.apply(field)

    assert output.grid == field.grid
    torch.testing.assert_close(
        output.data,
        before * torch.tensor(0.5 + 0.25j, dtype=before.dtype),
    )
    torch.testing.assert_close(field.data, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "layer",
    [
        IncidentSource.plane_wave(_grid(), 500e-9),
        ApertureLayer(
            ApertureSpec(shape=ApertureShape.CIRCLE, size_x_m=3e-6)
        ),
        IdealLensLayer(focal_length_m=50e-6),
        ComplexMaskLayer(
            transmission=torch.ones(_grid().shape, dtype=torch.complex128),
            grid=_grid(),
        ),
    ],
)
def test_layer_configs_are_strict_json_and_hashes_are_deterministic(layer) -> None:
    encoded = json.dumps(layer.to_config(), sort_keys=True, allow_nan=False)

    assert json.loads(encoded)["id"] == str(layer.id)
    assert len(layer.parameter_hash) == 64
    assert layer.parameter_hash == layer.parameter_hash


def test_source_lens_mask_multilayer_chain_matches_manual_application() -> None:
    grid = _grid()
    seed = _field(grid=grid)
    source = IncidentSource.gaussian(
        grid,
        seed.wavelength_m,
        waist_x_m=3e-6,
    )
    lens = IdealLensLayer(
        focal_length_m=60e-6,
        design_wavelength_m=seed.wavelength_m,
    )
    mask = ComplexMaskLayer(
        transmission=torch.full(
            grid.shape,
            0.8 + 0.1j,
            dtype=torch.complex128,
        ),
        grid=grid,
    )

    source_field = source.create(dtype=seed.dtype)
    result = OpticalSystem(layers=(lens, mask)).execute(source_field)
    manual = mask.apply(lens.apply(source_field))

    torch.testing.assert_close(result.output_field.data, manual.data)
    assert tuple(item.layer_id for item in result.layer_results) == (
        lens.id,
        mask.id,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_source_and_thin_layers_preserve_cuda() -> None:
    grid = _grid()
    incoming = _field(grid=grid, dtype=torch.complex64, device="cuda")
    source = IncidentSource.gaussian(
        grid,
        incoming.wavelength_m,
        waist_x_m=3e-6,
    )
    lens = IdealLensLayer(focal_length_m=50e-6)
    mask = ComplexMaskLayer(
        transmission=torch.ones(grid.shape, dtype=torch.complex64),
        grid=grid,
    )

    source_field = source.create(device="cuda", dtype=torch.complex64)
    result = OpticalSystem(layers=(lens, mask)).execute(source_field)

    assert result.output_field.device.type == "cuda"
    assert result.output_field.dtype == torch.complex64
