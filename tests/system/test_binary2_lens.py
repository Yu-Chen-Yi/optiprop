"""Lock Binary2 units, phase sign, serialization and shared Ex/Ey response."""

from dataclasses import replace
import math

import pytest
import torch

from optiprop import Binary2LensLayer, Binary2Phase, Field2D, Grid2D, NearField
from optiprop.project import OptiPropProject, load_project, project_to_domain, save_project
from optiprop.propagation import PaddingSpec, PropagationSpec
from optiprop.system import ApertureSpec, IdealLensLayer, IncidentSource, OpticalSystem, PropagationLayer


def incoming(dtype=torch.complex128, device="cpu"):
    grid = Grid2D(11, 9, 2e-6, 3e-6, 1e-6, -2e-6)
    return IncidentSource.plane_wave(
        grid, 1.31e-6, components=("Ex", "Ey"), amplitudes=(1, .5j)
    ).create(dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_mm_even_power_formula_and_shared_components(dtype):
    source = incoming(dtype)
    lens = Binary2LensLayer(
        [-12000, 400000, -2e7], phase_offset_rad=.4,
        transmission_amplitude=.7, center_x_m=3e-6, center_y_m=-1e-6,
        aperture=ApertureSpec("circle", 21e-6, center_x_m=3e-6, center_y_m=-1e-6),
    )
    x, y = source.grid.meshgrid(dtype=source.real_dtype)
    r2 = ((x - 3e-6) * 1000) ** 2 + ((y + 1e-6) * 1000) ** 2
    phase = .4 - 12000 * r2 + 400000 * r2**2 - 2e7 * r2**3
    expected = .7 * torch.exp(1j * phase) * lens.aperture.mask(source.grid, dtype=source.real_dtype)
    result = lens.apply(source)
    torch.testing.assert_close(result.data, source.data * expected)
    assert result.dtype == dtype
    assert result.grid == source.grid and result.z_m == source.z_m
    torch.testing.assert_close(result.data[1], .5j * result.data[0])


def test_matches_legacy_binary2_phase_on_the_same_grid():
    near = NearField(pixel_size=1e-6, field_Lx=20e-6, field_Ly=16e-6, dtype=torch.float64)
    legacy = Binary2Phase(near)
    expected = legacy.calculate_phase(
        binary2=[-10000, 2e6, -3e7, 4e8], lens_center=[2e-6, -1e-6],
        aperture_size=[9.3e-6], amplitude=.8, phase_offset=.3,
    )
    grid = Grid2D(near.X.shape[1], near.X.shape[0], 1e-6, 1e-6)
    source = IncidentSource.plane_wave(grid, 1.31e-6).create(dtype=torch.complex128)
    current = Binary2LensLayer(
        [-10000, 2e6, -3e7, 4e8], center_x_m=2e-6, center_y_m=-1e-6,
        aperture=ApertureSpec("circle", 9.3e-6, center_x_m=2e-6, center_y_m=-1e-6),
        transmission_amplitude=.8, phase_offset_rad=.3,
    )
    torch.testing.assert_close(current.evaluate_transmission(source), expected)


def test_first_coefficient_matches_paraxial_positive_focusing_sign():
    source = incoming()
    focal_length = 100e-6
    c1 = -math.pi * 1e-6 / (source.wavelength_m * focal_length)
    binary = Binary2LensLayer([c1])
    ideal = IdealLensLayer(focal_length, phase_model="paraxial")
    torch.testing.assert_close(binary.evaluate_transmission(source), ideal.evaluate_transmission(source))


def test_phase_is_fixed_and_does_not_invent_wavelength_or_medium_scaling():
    source = incoming()
    lens = Binary2LensLayer([-10000, 20])
    changed = replace(source, wavelength_m=1.55e-6, medium_index=1.5)
    torch.testing.assert_close(lens.apply(source).data, lens.apply(changed).data)


def test_coefficients_snapshot_hash_and_zero_phase():
    values = [-10000, 0]
    lens = Binary2LensLayer(values)
    values[0] = 0
    assert lens.coefficients == (-10000, 0)
    assert lens.parameter_hash != Binary2LensLayer(values).parameter_hash
    source = incoming()
    torch.testing.assert_close(Binary2LensLayer([0]).apply(source).data, source.data)
    torch.testing.assert_close(replace(lens, enabled=False).apply(source).data, source.data)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [True], "1,2", 1, [[1]]])
def test_bad_coefficients_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        Binary2LensLayer(values)


@pytest.mark.parametrize("kwargs", [
    {"transmission_amplitude": -1}, {"transmission_amplitude": 1.1},
    {"center_x_m": float("inf")}, {"phase_offset_rad": float("nan")}, {"aperture": {}},
])
def test_bad_common_parameters_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Binary2LensLayer([1], **kwargs)


def test_overflow_raises_instead_of_returning_nan():
    with pytest.raises(ValueError, match="overflow"):
        Binary2LensLayer([1e300]).apply(incoming(torch.complex64))


def test_preserves_input_field_autograd():
    base = incoming()
    data = base.data.clone().requires_grad_()
    field = base.with_data(data)
    output = Binary2LensLayer([-10000, 100]).apply(field)
    output.data.real.sum().backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert data.grad.abs().sum() > 0


def test_project_json_roundtrip_and_real_asm_propagation(tmp_path):
    source = IncidentSource.plane_wave(Grid2D(32, 24, 1e-6, 1e-6), 1.31e-6,
                                      components=("Ex", "Ey"), amplitudes=(1, .5))
    lens = Binary2LensLayer([-10000, 200], aperture=ApertureSpec("circle", 20e-6))
    segment = PropagationLayer(PropagationSpec("asm", 30e-6, padding=PaddingSpec.by_factor(2)))
    system = OpticalSystem((lens, segment))
    project = OptiPropProject.from_domain(source, system)
    path = tmp_path / "binary2.json"
    save_project(project, path)
    loaded = load_project(path)
    restored_source, restored = project_to_domain(loaded)
    assert loaded.to_dict() == project.to_dict()
    assert restored.layers[0].id == lens.id
    assert restored.layers[1].spec.distance_m == segment.spec.distance_m
    expected = system.execute(source.create(dtype=torch.complex128)).output_field
    actual = restored.execute(restored_source.create(dtype=torch.complex128)).output_field
    torch.testing.assert_close(actual.data, expected.data)
    assert torch.isfinite(actual.data).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_dtype_and_cpu_agree():
    layer = Binary2LensLayer([-10000, 200])
    result = layer.apply(incoming(torch.complex64, "cuda"))
    assert result.device.type == "cuda"
    torch.testing.assert_close(result.data.cpu(), layer.apply(incoming(torch.complex64)).data)
