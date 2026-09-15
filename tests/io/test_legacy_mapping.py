"""Explicit legacy-field mapping, unit, and orientation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from optiprop.io import (
    AmbiguousMappingError,
    AmplitudeConvention,
    AxisOrder,
    ImportMapping,
    ImportMappingError,
    LengthUnit,
    MappingConfidence,
    PhaseUnit,
    inspect_field,
    load_field,
)


def _save_payload(path: Path, payload: dict[str, object]) -> None:
    if path.suffix == ".npz":
        np.savez(path, **payload)
    else:
        scipy_io = pytest.importorskip("scipy.io")
        scipy_io.savemat(path, payload)


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_noncanonical_aliases_are_inspected_but_never_silently_loaded(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"legacy{suffix}"
    data = np.arange(15, dtype=np.float64).reshape(3, 5).astype(np.complex128)
    _save_payload(
        path,
        {
            "U": data,
            "dx": np.array([[0.7]]),
            "dy": np.array([[1.2]]),
            "wavelength": np.array([[532.0]]),
        },
    )

    inspection = inspect_field(path)

    assert not inspection.is_canonical
    assert inspection.mapping_confidence in {
        MappingConfidence.SUGGESTED,
        MappingConfidence.AMBIGUOUS,
    }
    assert inspection.suggested_mapping is not None
    with pytest.raises(AmbiguousMappingError):
        load_field(path)


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_explicit_mapping_separates_spatial_and_wavelength_units(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"units{suffix}"
    data = (
        np.arange(15, dtype=np.float64).reshape(3, 5)
        + 1j * np.arange(15, dtype=np.float64).reshape(3, 5)[::-1]
    )
    _save_payload(
        path,
        {
            "U": data,
            "dx": 0.7,
            "dy": 1.2,
            "lambda": 532.0,
        },
    )
    mapping = ImportMapping(
        data_keys=("U",),
        components=("scalar",),
        dx_key="dx",
        dy_key="dy",
        wavelength_key="lambda",
        spatial_unit=LengthUnit.UM,
        wavelength_unit=LengthUnit.NM,
    )

    field = load_field(path, mapping)

    assert field.grid.shape == (3, 5)
    assert field.grid.dx == pytest.approx(0.7e-6)
    assert field.grid.dy == pytest.approx(1.2e-6)
    assert field.wavelength_m == pytest.approx(532e-9)
    torch.testing.assert_close(
        field.data[0],
        torch.from_numpy(data),
    )


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_ex_ey_axis_transpose_and_flips_are_explicit(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"axes{suffix}"
    ex = np.arange(15, dtype=np.float64).reshape(3, 5).astype(np.complex128)
    ey = (100 + np.arange(15, dtype=np.float64)).reshape(3, 5).astype(
        np.complex128
    )
    _save_payload(
        path,
        {
            "EX": ex.T,
            "EY": ey.T,
            "pixel_size": 0.8,
        },
    )
    mapping = ImportMapping(
        data_keys=("EX", "EY"),
        components=("Ex", "Ey"),
        pixel_size_key="pixel_size",
        wavelength_m=633e-9,
        spatial_unit=LengthUnit.UM,
        axis_order=AxisOrder.XY,
        flip_x=True,
        flip_y=True,
    )

    field = load_field(path, mapping)

    expected = torch.flip(
        torch.from_numpy(np.stack((ex, ey))),
        dims=(-2, -1),
    )
    assert field.grid.shape == (3, 5)
    torch.testing.assert_close(field.data, expected)


@pytest.mark.parametrize("phase_unit", [PhaseUnit.RAD, PhaseUnit.DEG])
def test_amplitude_phase_mapping_with_intensity_convention(
    tmp_path: Path,
    phase_unit: PhaseUnit,
) -> None:
    path = tmp_path / f"amplitude_{phase_unit.value}.npz"
    intensity = np.array(
        [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]],
        dtype=np.float64,
    )
    phase_rad = np.array(
        [[0.0, np.pi / 2, np.pi], [-np.pi / 2, 0.25, -0.5]],
        dtype=np.float64,
    )
    phase = np.rad2deg(phase_rad) if phase_unit is PhaseUnit.DEG else phase_rad
    np.savez(path, intensity=intensity, phase=phase)
    mapping = ImportMapping(
        amplitude_keys=("intensity",),
        phase_keys=("phase",),
        components=("scalar",),
        dx_m=1e-6,
        dy_m=2e-6,
        wavelength_m=500e-9,
        amplitude_convention=AmplitudeConvention.INTENSITY,
        phase_unit=phase_unit,
    )

    field = load_field(path, mapping)
    expected = np.sqrt(intensity) * np.exp(1j * phase_rad)

    torch.testing.assert_close(
        field.data[0],
        torch.from_numpy(expected),
    )


def test_missing_wavelength_or_ambiguous_unit_is_never_guessed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_wavelength.npz"
    np.savez(path, U=np.ones((3, 5), dtype=np.complex128), dx=1.0, dy=1.0)

    missing_wavelength = ImportMapping(
        data_keys=("U",),
        dx_key="dx",
        dy_key="dy",
        spatial_unit=LengthUnit.UM,
    )
    missing_units = ImportMapping(
        data_keys=("U",),
        dx_key="dx",
        dy_key="dy",
        wavelength_m=500e-9,
    )

    with pytest.raises((AmbiguousMappingError, ValueError)):
        load_field(path, missing_wavelength)
    with pytest.raises((AmbiguousMappingError, ValueError)):
        load_field(path, missing_units)


def test_explicit_scalar_overrides_need_no_metadata_keys(tmp_path: Path) -> None:
    path = tmp_path / "bare.npz"
    data = np.ones((4, 7), dtype=np.complex64) * (0.5 + 0.25j)
    np.savez(path, arbitrary_name=data)
    mapping = ImportMapping(
        data_keys=("arbitrary_name",),
        dx_m=0.5e-6,
        dy_m=0.75e-6,
        wavelength_m=1064e-9,
        medium_index=1.33 + 0.01j,
        x_center_m=2e-6,
        y_center_m=-3e-6,
        z_m=8e-6,
    )

    field = load_field(path, mapping)

    assert field.grid.shape == (4, 7)
    assert field.grid.x_center == pytest.approx(2e-6)
    assert field.grid.y_center == pytest.approx(-3e-6)
    assert field.medium_index == pytest.approx(1.33 + 0.01j)
    assert field.z_m == pytest.approx(8e-6)


def test_explicit_nx_ny_metadata_must_match_oriented_field(tmp_path: Path) -> None:
    path = tmp_path / "bad_dimensions.npz"
    np.savez(
        path,
        U=np.ones((3, 5), dtype=np.complex128),
        nx=np.array([[3]]),
        ny=np.array([[5]]),
    )
    mapping = ImportMapping(
        data_keys=("U",),
        dx_m=1e-6,
        dy_m=1e-6,
        wavelength_m=500e-9,
        nx_key="nx",
        ny_key="ny",
    )

    with pytest.raises(ImportMappingError, match="nx|ny|dimension"):
        load_field(path, mapping)
