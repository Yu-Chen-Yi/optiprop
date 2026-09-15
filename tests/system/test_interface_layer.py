"""Physics and system-integration tests for homogeneous-medium interfaces."""

from __future__ import annotations

import json

import pytest
import torch

from optiprop import Field2D, Grid2D
from optiprop.propagation import PropagationMethod, PropagationSpec
from optiprop.system.interface import InterfaceLayer
from optiprop.system.layers import PropagationLayer
from optiprop.system.optical_system import OpticalSystem, SystemValidationError


def _field(
    *,
    medium_index: complex = 1.0,
    components: tuple[str, ...] = ("scalar",),
    device: str = "cpu",
    requires_grad: bool = False,
) -> Field2D:
    grid = Grid2D(nx=9, ny=7, dx=1.0e-6, dy=1.2e-6)
    return Field2D(
        data=torch.ones(
            (len(components), grid.ny, grid.nx),
            dtype=torch.complex128,
            device=device,
            requires_grad=requires_grad,
        ),
        grid=grid,
        wavelength_m=633e-9,
        medium_index=medium_index,
        components=components,
    )


@pytest.mark.parametrize("components", [("scalar",), ("Ex", "Ey")])
def test_interface_applies_normal_incidence_fresnel_to_all_components(
    components: tuple[str, ...],
) -> None:
    field = _field(medium_index=1.0, components=components)
    layer = InterfaceLayer(n1=1.0, n2=1.5)

    output = layer.apply(field)

    expected_t = 2.0 / 2.5
    torch.testing.assert_close(
        output.data,
        field.data * expected_t,
        rtol=0,
        atol=0,
    )
    assert output.medium_index == 1.5 + 0j
    assert output.grid is field.grid
    assert output.z_m == field.z_m
    assert layer.transmission_coefficient == pytest.approx(expected_t)
    assert layer.reflection_coefficient == pytest.approx(-0.2)


def test_ignore_reflection_updates_medium_without_amplitude_change() -> None:
    field = _field(medium_index=1.0)
    layer = InterfaceLayer(n1=1.0, n2=1.45, ignore_reflection=True)

    report = layer.validate(field)
    output = layer.apply(field)

    assert report.is_valid
    assert report.warnings[0].code == "interface.reflection_ignored"
    assert report.warnings[0].layer_id == str(layer.id)
    torch.testing.assert_close(output.data, field.data, rtol=0, atol=0)
    assert output.medium_index == 1.45 + 0j


def test_interface_rejects_incoming_medium_mismatch_with_layer_path() -> None:
    field = _field(medium_index=1.33)
    layer = InterfaceLayer(n1=1.0, n2=1.5)

    report = layer.validate(field)

    assert not report.is_valid
    assert report.errors[0].code == "interface.incident_index_mismatch"
    assert report.errors[0].layer_id == str(layer.id)
    assert report.errors[0].parameter_path == "n1"
    with pytest.raises(SystemValidationError) as exc_info:
        OpticalSystem(layers=(layer,)).execute(field)
    assert exc_info.value.report.errors[0].layer_id == str(layer.id)


def test_interface_preview_updates_downstream_propagation_medium() -> None:
    field = _field(medium_index=1.0)
    interface = InterfaceLayer(n1=1.0, n2=1.5)
    propagation = PropagationLayer(
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=5e-6,
            medium_index=1.5,
        )
    )
    system = OpticalSystem(layers=(interface, propagation))

    report = system.validate(field)
    result = system.execute(field)

    assert report.is_valid
    assert result.layer_results[0].output_field.medium_index == 1.5 + 0j
    assert result.final_field.medium_index == 1.5 + 0j
    assert result.final_field.z_m == pytest.approx(5e-6)


def test_propagation_medium_change_requires_explicit_interface() -> None:
    field = _field(medium_index=1.0)
    propagation = PropagationLayer(
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=5e-6,
            medium_index=1.5,
        )
    )

    report = OpticalSystem(layers=(propagation,)).validate(field)

    assert not report.is_valid
    assert report.errors[0].code == "medium.interface_required"
    assert report.errors[0].layer_id == str(propagation.id)
    assert report.errors[0].parameter_path == "spec.medium_index"


def test_disabled_interface_is_exact_identity() -> None:
    field = _field(medium_index=1.0)
    layer = InterfaceLayer(n1=1.0, n2=1.5, enabled=False)

    assert layer.apply(field) is field
    result = OpticalSystem(layers=(layer,)).execute(field)
    assert result.final_field is field
    assert result.layer_results[0].output_field is field


def test_interface_supports_complex_index_and_preserves_autograd() -> None:
    field = _field(medium_index=1.0, requires_grad=True)
    layer = InterfaceLayer(n1=1.0, n2=1.5 + 0.02j)

    output = layer.apply(field)
    output.intensity().sum().backward()

    assert output.medium_index == 1.5 + 0.02j
    assert field.data.grad is not None
    assert torch.isfinite(field.data.grad).all()


def test_interface_rejects_active_medium() -> None:
    field = _field(medium_index=1.0)
    layer = InterfaceLayer(n1=1.0, n2=1.5 - 0.02j)

    report = layer.validate(field)

    assert not report.is_valid
    issue = next(
        item for item in report.errors if item.code == "interface.active_medium"
    )
    assert issue.layer_id == str(layer.id)
    assert issue.parameter_path == "n2"


def test_interface_config_and_parameter_hash_are_json_native() -> None:
    first = InterfaceLayer(n1=1.0, n2=1.5 + 0.01j, name="first")
    second = InterfaceLayer(n1=1.0, n2=1.5 + 0.01j, name="second")

    encoded = json.dumps(first.to_config(), sort_keys=True, allow_nan=False)

    assert json.loads(encoded)["n2"] == {"real": 1.5, "imag": 0.01}
    assert first.parameter_hash == second.parameter_hash


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_interface_preserves_cuda() -> None:
    field = _field(medium_index=1.0, device="cuda")

    output = InterfaceLayer(n1=1.0, n2=1.5).apply(field)

    assert output.device.type == "cuda"
    assert output.dtype == field.dtype
