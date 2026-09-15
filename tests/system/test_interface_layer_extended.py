"""Additional analytic/provenance coverage for InterfaceLayer."""

from __future__ import annotations

import pytest
import torch

from optiprop import Field2D, Grid2D
from optiprop.propagation import PaddingSpec, PropagationMethod, PropagationSpec
from optiprop.system.interface import InterfaceLayer, InterfaceModel
from optiprop.system.layers import PropagationLayer
from optiprop.system.optical_system import OpticalSystem


def _field(*, medium_index: complex = 1.0) -> Field2D:
    grid = Grid2D(nx=7, ny=5, dx=1e-6, dy=1.5e-6)
    return Field2D(
        data=torch.ones((1, grid.ny, grid.nx), dtype=torch.complex128),
        grid=grid,
        wavelength_m=633e-9,
        medium_index=medium_index,
        z_m=4e-6,
    )


def test_model_and_warnings_do_not_claim_oblique_or_reflected_branches() -> None:
    field = _field()
    layer = InterfaceLayer(n1=1.0, n2=1.5)

    report = layer.validate(field)
    codes = {issue.code for issue in report}

    assert layer.model is InterfaceModel.NORMAL_INCIDENCE_SCALAR
    assert "interface.normal_incidence_scalar_model" in codes
    assert "interface.reflected_field_discarded" in codes
    assert any(
        "s/p" in issue.message for issue in report.info
    )
    assert any(
        "multiple reflections" in issue.message
        for issue in report.warnings
    )


def test_real_index_power_coefficients_conserve_energy() -> None:
    n1 = 1.0
    n2 = 1.7
    layer = InterfaceLayer(n1=n1, n2=n2)

    reflected_power = abs(layer.reflection_coefficient) ** 2
    transmitted_power = n2 / n1 * abs(layer.transmission_coefficient) ** 2

    assert reflected_power + transmitted_power == pytest.approx(1.0, abs=1e-15)


def test_complex_passive_indices_stay_finite_without_false_conservation_claim() -> None:
    field = _field(medium_index=1.0 + 0.02j)
    layer = InterfaceLayer(n1=1.0 + 0.02j, n2=1.5 + 0.1j)

    report = layer.validate(field)
    output = layer.apply(field)

    assert report.is_valid
    assert torch.isfinite(output.data.real).all()
    assert torch.isfinite(output.data.imag).all()
    assert not any(
        "conservation" in issue.code or "conservation" in issue.message.lower()
        for issue in report
    )


def test_air_glass_air_chain_matches_manual_single_pass_coefficients() -> None:
    field = _field()
    enter = InterfaceLayer(n1=1.0, n2=1.5, name="enter glass")
    medium = PropagationLayer(
        name="glass",
        spec=PropagationSpec(
            method=PropagationMethod.ASM,
            distance_m=0.0,
            medium_index=1.5,
            padding=PaddingSpec.none(),
        ),
    )
    leave = InterfaceLayer(n1=1.5, n2=1.0, name="leave glass")

    result = OpticalSystem(layers=(enter, medium, leave)).execute(field)
    expected_coefficient = (
        enter.transmission_coefficient * leave.transmission_coefficient
    )

    torch.testing.assert_close(
        result.output_field.data,
        field.data * expected_coefficient,
    )
    assert result.output_field.medium_index == 1.0 + 0j
    assert result.output_field.z_m == field.z_m
