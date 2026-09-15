"""Smoke tests that protect the public OptiProp 1.x API during the 2.0 work."""

import pytest
import torch

import optiprop


def test_legacy_near_field_and_lens_still_work():
    field = optiprop.NearField(
        pixel_size=1e-6,
        field_Lx=8e-6,
        field_Ly=6e-6,
        device="cpu",
    )
    lens = optiprop.EqualPathPhase(field)

    transmission = lens.calculate_phase(
        focal_length=20e-6,
        design_lambda=0.94e-6,
        lens_diameter=6e-6,
    )

    assert transmission.shape == (field.Nx, field.Ny)
    assert torch.is_complex(transmission)
    assert bool(torch.isfinite(transmission).all())


def test_legacy_fresnel_propagation_smoke():
    field = optiprop.NearField(
        pixel_size=1e-6,
        field_Lx=8e-6,
        field_Ly=6e-6,
        device="cpu",
    )
    source = optiprop.IncidentField(field).calculate_phase(
        design_lambda=0.94e-6,
        aperture_type="rectangle",
        aperture_size=[8e-6, 6e-6],
    )
    propagator = optiprop.FresnelPropagation(
        propagation_wavelength=0.94e-6,
        propagation_distance=20e-6,
        device="cpu",
    )

    propagator.set_input_field(source, field.pixel_size)
    propagator.propagate()

    assert propagator.get_output_U.shape == source.shape
    assert torch.is_complex(propagator.get_output_U)
    assert bool(torch.isfinite(propagator.get_output_U).all())


def test_legacy_fresnel_xz_scan_uses_rectangular_x_axis():
    field = optiprop.NearField(
        pixel_size=1e-6,
        field_Lx=8e-6,
        field_Ly=6e-6,
        device="cpu",
    )
    source = optiprop.IncidentField(field).calculate_phase(
        design_lambda=0.94e-6,
        aperture_type="rectangle",
        aperture_size=[8e-6, 6e-6],
    )
    propagator = optiprop.FresnelPropagation(
        propagation_wavelength=0.94e-6,
        propagation_distance=20e-6,
        device="cpu",
    )
    propagator.set_input_field(source, field.pixel_size)

    propagator.propagate_xz(z_range=[20e-6, 30e-6])

    assert propagator.get_output_UZ.shape == (2, source.shape[1])
    assert propagator.get_output_X.shape == propagator.get_output_UZ.shape
    assert bool(torch.isfinite(propagator.get_output_UZ).all())


def test_legacy_rayleigh_sommerfeld_small_rectangular_output_smoke():
    field = optiprop.NearField(
        pixel_size=1e-6,
        field_Lx=4e-6,
        field_Ly=3e-6,
        device="cpu",
    )
    source = optiprop.IncidentField(field).calculate_phase(
        design_lambda=0.94e-6,
        aperture_type="rectangle",
        aperture_size=[4e-6, 3e-6],
    )
    propagator = optiprop.RayleighSommerfeldPropagation(
        propagation_wavelength=0.94e-6,
        propagation_distance=8e-6,
        device="cpu",
    )
    propagator.set_input_field(source, field.pixel_size)
    propagator.set_output_field(
        output_pixel_size=1.2e-6,
        output_size=[3, 2],
        center=[0.3e-6, -0.2e-6],
    )

    propagator.propagate()

    assert propagator.get_output_U.shape == (3, 2)
    assert torch.is_complex(propagator.get_output_U)
    assert bool(torch.isfinite(propagator.get_output_U).all())


def test_package_version_is_exposed():
    assert isinstance(optiprop.__version__, str)
    assert optiprop.__version__


def test_canonical_field_model_is_exported_without_removing_legacy_api():
    assert optiprop.Grid2D is not None
    assert optiprop.Field2D is not None
    assert optiprop.NearField is not None
