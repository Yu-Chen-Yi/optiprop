"""Canonical NPZ/MAT round-trip and inspection tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from optiprop import Field2D, Grid2D
from optiprop.io import (
    FieldFormat,
    FieldIOError,
    UnsupportedFormatError,
    inspect_field,
    load_field,
    save_field,
)


def _field(
    *,
    components: tuple[str, ...],
    dtype: torch.dtype,
    device: str = "cpu",
) -> Field2D:
    grid = Grid2D(
        nx=8,
        ny=5,
        dx=0.7e-6,
        dy=1.3e-6,
        x_center=2.1e-6,
        y_center=-1.4e-6,
    )
    real_dtype = (
        torch.float32 if dtype == torch.complex64 else torch.float64
    )
    x, y = grid.meshgrid(dtype=real_dtype, device=device)
    first = (1 + 0.2 * x / grid.dx) * torch.exp(
        1j * (0.17 * x / grid.dx - 0.23 * y / grid.dy)
    )
    data = first[None]
    if components == ("Ex", "Ey"):
        second = (0.4 - 0.03 * y / grid.dy) * torch.exp(
            1j * (-0.11 * x / grid.dx + 0.31 * y / grid.dy)
        )
        data = torch.stack((first, second))
    return Field2D(
        data=data.to(dtype),
        grid=grid,
        wavelength_m=532e-9,
        medium_index=1.45 + 0.008j,
        components=components,
        z_m=13e-6,
        metadata={
            "source": "pytest",
            "run": 7,
            "nested": {"labels": ["rectangular", "roundtrip"]},
        },
    )


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
@pytest.mark.parametrize(
    ("components", "dtype"),
    [
        (("scalar",), torch.complex64),
        (("scalar",), torch.complex128),
        (("Ex", "Ey"), torch.complex64),
        (("Ex", "Ey"), torch.complex128),
    ],
)
def test_canonical_rectangular_round_trip(
    tmp_path: Path,
    suffix: str,
    components: tuple[str, ...],
    dtype: torch.dtype,
) -> None:
    field = _field(components=components, dtype=dtype)
    path = tmp_path / f"canonical{suffix}"

    report = save_field(field, path)
    loaded = load_field(path)

    assert path.is_file()
    assert report.path == path
    assert loaded.grid == field.grid
    assert loaded.components == field.components
    assert loaded.dtype == field.dtype
    assert loaded.wavelength_m == pytest.approx(field.wavelength_m)
    assert loaded.medium_index == pytest.approx(field.medium_index)
    assert loaded.z_m == pytest.approx(field.z_m)
    torch.testing.assert_close(loaded.data, field.data.cpu(), rtol=0, atol=0)
    assert loaded.metadata["source"] == "pytest"
    assert loaded.metadata["run"] == 7
    assert tuple(loaded.metadata["nested"]["labels"]) == (
        "rectangular",
        "roundtrip",
    )


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_canonical_saved_keys_and_json_metadata(
    tmp_path: Path,
    suffix: str,
) -> None:
    field = _field(components=("Ex", "Ey"), dtype=torch.complex128)
    path = tmp_path / f"keys{suffix}"
    save_field(field, path)

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            metadata = json.loads(str(payload["metadata_json"].item()))
    else:
        scipy_io = pytest.importorskip("scipy.io")
        payload = scipy_io.loadmat(path, squeeze_me=True)
        keys = {key for key in payload if not key.startswith("__")}
        metadata_value = payload["metadata_json"]
        metadata = json.loads(str(metadata_value.item() if hasattr(
            metadata_value, "item"
        ) else metadata_value))

    assert {
        "schema_name",
        "schema_version",
        "field",
        "components",
        "dx_m",
        "dy_m",
        "wavelength_m",
        "medium_index_real",
        "medium_index_imag",
        "x_center_m",
        "y_center_m",
        "z_m",
        "metadata_json",
    } <= keys
    assert metadata["source"] == "pytest"


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_canonical_inspection_is_strict_and_non_mutating(
    tmp_path: Path,
    suffix: str,
) -> None:
    field = _field(components=("scalar",), dtype=torch.complex128)
    before = field.data.clone()
    path = tmp_path / f"inspect{suffix}"
    save_field(field, path)

    inspection = inspect_field(path)

    assert inspection.path == path
    assert inspection.format in {FieldFormat.NPZ, FieldFormat.MAT}
    assert inspection.is_canonical
    assert inspection.schema_name == "optiprop.field2d"
    assert "field" in inspection.keys
    field_info = next(item for item in inspection.arrays if item.key == "field")
    assert field_info.shape in {(5, 8), (1, 5, 8)}
    assert field_info.is_complex
    assert field_info.magnitude_min is not None
    assert field_info.magnitude_max is not None
    torch.testing.assert_close(field.data, before, rtol=0, atol=0)


def test_cuda_export_moves_only_serialized_copy_to_cpu(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    field = _field(
        components=("Ex", "Ey"),
        dtype=torch.complex64,
        device="cuda",
    )
    before = field.data.clone()
    path = tmp_path / "cuda.npz"

    save_field(field, path)
    loaded = load_field(path)

    assert field.device.type == "cuda"
    torch.testing.assert_close(field.data, before, rtol=0, atol=0)
    assert loaded.device.type == "cpu"
    assert loaded.dtype == torch.complex64
    torch.testing.assert_close(loaded.data, field.data.cpu(), rtol=0, atol=0)


def test_explicit_format_must_agree_with_supported_output(tmp_path: Path) -> None:
    field = _field(components=("scalar",), dtype=torch.complex128)

    with pytest.raises(UnsupportedFormatError):
        save_field(field, tmp_path / "field.bin")


@pytest.mark.parametrize(
    "bad_metadata",
    [
        {"callable": lambda: None},
        {"nonfinite": float("nan")},
        {"object": object()},
    ],
)
@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_atomic_export_failure_preserves_existing_target(
    tmp_path: Path,
    bad_metadata: dict[str, object],
    suffix: str,
) -> None:
    path = tmp_path / f"existing{suffix}"
    original_bytes = b"existing user data must survive"
    path.write_bytes(original_bytes)
    field = replace(
        _field(components=("scalar",), dtype=torch.complex128),
        metadata=bad_metadata,
    )

    with pytest.raises((FieldIOError, TypeError, ValueError)):
        save_field(field, path)

    assert path.read_bytes() == original_bytes


def test_highly_compressible_writer_falls_back_to_safe_self_round_trip(
    tmp_path: Path,
) -> None:
    grid = Grid2D(nx=512, ny=512, dx=1e-6, dy=1e-6)
    field = Field2D(
        data=torch.zeros((1, grid.ny, grid.nx), dtype=torch.complex64),
        grid=grid,
        wavelength_m=633e-9,
    )
    path = tmp_path / "highly_compressible.npz"

    report = save_field(field, path)
    loaded = load_field(path)

    assert not report.compressed
    assert any(
        warning.code == "io.compression_fallback"
        for warning in report.warnings
    )
    torch.testing.assert_close(loaded.data, field.data, rtol=0, atol=0)
