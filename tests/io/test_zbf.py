"""Strict ZBF codec and public-dispatch tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from optiprop import Field2D, Grid2D
from optiprop.io import (
    ExportOptions,
    FieldFormat,
    ImportMapping,
    ImportMappingError,
    SchemaValidationError,
    ZbfBeam,
    ZbfPilotRays,
    inspect_field,
    load_field,
    read_zbf,
    save_field,
    write_zbf,
)


def _field(*, polarized: bool = False) -> Field2D:
    grid = Grid2D(nx=5, ny=3, dx=0.7e-6, dy=1.2e-6)
    values = torch.arange(15, dtype=torch.float64).reshape(3, 5)
    ex = torch.complex(values, values.flip(1) / 10)
    data = ex[None]
    components = ("scalar",)
    if polarized:
        data = torch.stack((ex, -0.25j * ex))
        components = ("Ex", "Ey")
    return Field2D(
        data=data, grid=grid, wavelength_m=532e-9,
        medium_index=1.45, components=components,
    )


@pytest.mark.parametrize("polarized", [False, True])
def test_public_dispatch_round_trip(tmp_path: Path, polarized: bool) -> None:
    field = _field(polarized=polarized)
    path = tmp_path / "field.zbf"

    report = save_field(field, path)
    inspected = inspect_field(path)
    loaded = load_field(path)

    assert report.format is FieldFormat.ZBF
    assert report.schema_version == 1
    assert report.compressed is False
    assert inspected.is_canonical
    assert inspected.scalar_metadata["dx_m"] == pytest.approx(field.grid.dx)
    assert loaded.grid == field.grid
    assert loaded.components == field.components
    torch.testing.assert_close(loaded.data, field.data)


def test_reader_converts_header_units_and_pilot_values_to_si(tmp_path: Path) -> None:
    beam = ZbfBeam(
        ex=np.ones((2, 3), dtype=np.complex128), dx_m=2e-3, dy_m=3e-3,
        wavelength_m=0.5e-3, medium_index=1.2,
        pilot=ZbfPilotRays(position_x_m=4e-3, waist_y_m=5e-3),
    )
    path = tmp_path / "metres.zbf"
    write_zbf(beam, path)
    payload = bytearray(path.read_bytes())
    # Convert all stored metre lengths to millimetres and select unit code 0.
    struct.pack_into("<i", payload, 16, 0)
    values = list(struct.unpack_from("<20d", payload, 36))
    for index in range(9):
        values[index] *= 1000
    struct.pack_into("<20d", payload, 36, *values)
    path.write_bytes(payload)

    loaded = read_zbf(path)
    assert loaded.dx_m == pytest.approx(beam.dx_m)
    assert loaded.wavelength_m == pytest.approx(beam.wavelength_m)
    assert loaded.pilot.position_x_m == pytest.approx(4e-3)


def test_v0_scalar_reader(tmp_path: Path) -> None:
    path = tmp_path / "old.zbf"
    ex = np.arange(6, dtype=np.float64).reshape(2, 3) * (1 + 0.5j)
    header = struct.pack("<9i", 0, 3, 2, 0, 0, 0, 0, 0, 0)
    values = struct.pack(
        "<11d", 0.01, 0.02, 0.03, 0.04, 0.0005, 0.06,
        0.0, 0.0, 0.0, 0.0, 0.0,
    )
    pairs = np.column_stack((ex.real.ravel(), ex.imag.ravel())).astype("<f8")
    path.write_bytes(header + values + pairs.tobytes())

    beam = read_zbf(path)
    assert beam.version == 0
    assert beam.medium_index == 1.0
    assert beam.dx_m == pytest.approx(10e-6)
    np.testing.assert_allclose(beam.ex, ex)


@pytest.mark.parametrize("cut", [1, 35, 36, 195, 196, 210])
def test_truncated_files_are_rejected(tmp_path: Path, cut: int) -> None:
    path = tmp_path / "truncated.zbf"
    write_zbf(
        ZbfBeam(ex=np.ones((2, 3), dtype=np.complex128), dx_m=1e-6,
                dy_m=1e-6, wavelength_m=500e-9),
        path,
    )
    path.write_bytes(path.read_bytes()[:cut])
    with pytest.raises(SchemaValidationError):
        read_zbf(path)


def test_trailing_bytes_are_strictly_rejected_but_can_be_ignored(tmp_path: Path) -> None:
    path = tmp_path / "trailing.zbf"
    write_zbf(
        ZbfBeam(ex=np.ones((1, 1), dtype=np.complex128), dx_m=1e-6,
                dy_m=1e-6, wavelength_m=500e-9),
        path,
    )
    path.write_bytes(path.read_bytes() + b"garbage")
    with pytest.raises(SchemaValidationError, match="trailing"):
        read_zbf(path)
    assert read_zbf(path, strict=False).ex.shape == (1, 1)


@pytest.mark.parametrize(
    ("offset", "value"),
    [(0, 2), (4, 0), (8, -1), (12, 2), (16, 9), (20, 1)],
)
def test_illegal_header_values_are_rejected(
    tmp_path: Path, offset: int, value: int,
) -> None:
    path = tmp_path / "invalid.zbf"
    write_zbf(
        ZbfBeam(ex=np.ones((1, 1), dtype=np.complex128), dx_m=1e-6,
                dy_m=1e-6, wavelength_m=500e-9),
        path,
    )
    payload = bytearray(path.read_bytes())
    struct.pack_into("<i", payload, offset, value)
    path.write_bytes(payload)
    with pytest.raises(SchemaValidationError):
        read_zbf(path)


def test_mapping_is_rejected_for_self_describing_zbf(tmp_path: Path) -> None:
    path = tmp_path / "field.zbf"
    save_field(_field(), path)
    with pytest.raises(ImportMappingError):
        load_field(path, ImportMapping(data_keys=("Ex",), dx_m=1, dy_m=1,
                                       wavelength_m=1))


def test_absorbing_medium_is_not_silently_lost(tmp_path: Path) -> None:
    field = replace(_field(), medium_index=1.5 + 0.01j)
    with pytest.raises(SchemaValidationError, match="real refractive index"):
        save_field(field, tmp_path / "lossy.zbf")


def test_atomic_replace_failure_preserves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "field.zbf"
    path.write_bytes(b"original")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("optiprop.io.zbf.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_field(_field(), path, options=ExportOptions(atomic=True))
    assert path.read_bytes() == b"original"
    assert list(tmp_path.glob(".field.zbf.*.tmp")) == []


def test_non_atomic_export_and_coordinate_warning(tmp_path: Path) -> None:
    base = _field()
    field = replace(
        base,
        grid=Grid2D(nx=5, ny=3, dx=0.7e-6, dy=1.2e-6, x_center=1e-6),
        z_m=2e-6,
    )
    report = save_field(
        field, tmp_path / "field.zbf", options=ExportOptions(atomic=False),
    )
    assert report.atomic is False
    assert [issue.code for issue in report.warnings] == [
        "io.zbf.unstored_coordinates"
    ]
