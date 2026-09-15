"""Malformed-schema and pickle/object-array safety tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from optiprop.io import (
    SchemaValidationError,
    UnsafeDatasetError,
    UnsupportedFormatError,
    inspect_field,
    load_field,
)


def _canonical_payload(
    *,
    field: np.ndarray | None = None,
    components: np.ndarray | None = None,
    metadata_json: object = "{}",
) -> dict[str, object]:
    return {
        "schema_name": np.array("optiprop.field2d"),
        "schema_version": np.array(1, dtype=np.int64),
        "field": (
            np.ones((3, 5), dtype=np.complex128)
            if field is None
            else field
        ),
        "components": (
            np.array(["scalar"], dtype="<U8")
            if components is None
            else components
        ),
        "dx_m": np.array(1e-6),
        "dy_m": np.array(2e-6),
        "wavelength_m": np.array(633e-9),
        "medium_index_real": np.array(1.0),
        "medium_index_imag": np.array(0.0),
        "x_center_m": np.array(0.0),
        "y_center_m": np.array(0.0),
        "z_m": np.array(0.0),
        "metadata_json": np.array(metadata_json),
    }


def test_npz_object_array_is_rejected_without_pickle_loading(tmp_path: Path) -> None:
    path = tmp_path / "object.npz"
    np.savez(
        path,
        field=np.array([{"code": "must never unpickle"}], dtype=object),
    )

    with pytest.raises(UnsafeDatasetError):
        inspect_field(path)
    with pytest.raises(UnsafeDatasetError):
        load_field(path)


def test_canonical_npz_never_requires_object_components_or_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "object_metadata.npz"
    payload = _canonical_payload()
    payload["metadata_json"] = np.array({"unsafe": "object"}, dtype=object)
    np.savez(path, **payload)

    with pytest.raises(UnsafeDatasetError):
        inspect_field(path)
    with pytest.raises(UnsafeDatasetError):
        load_field(path)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def test_npz_duplicate_member_name_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    member = _npy_bytes(np.ones((2, 3), dtype=np.complex128))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("field.npy", member)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("field.npy", member)

    with pytest.raises(UnsafeDatasetError):
        inspect_field(path)


def test_npz_path_traversal_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "traversal.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "../field.npy",
            _npy_bytes(np.ones((2, 3), dtype=np.complex128)),
        )

    with pytest.raises(UnsafeDatasetError):
        inspect_field(path)


def test_npz_extreme_compression_ratio_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "compression_bomb.npz"
    zeros = np.zeros((2048, 2048), dtype=np.float64)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr("field.npy", _npy_bytes(zeros))

    with pytest.raises(UnsafeDatasetError):
        inspect_field(path)


@pytest.mark.parametrize(
    ("name", "update"),
    [
        ("missing_dy", {"dy_m": None}),
        ("future_schema", {"schema_version": np.array(999)}),
        ("invalid_json", {"metadata_json": np.array("{broken")}),
        (
            "real_field",
            {"field": np.ones((3, 5), dtype=np.float64)},
        ),
        (
            "component_shape",
            {
                "field": np.ones((2, 3, 5), dtype=np.complex128),
                "components": np.array(["scalar"], dtype="<U8"),
            },
        ),
        (
            "nan_spacing",
            {"dx_m": np.array(float("nan"))},
        ),
    ],
)
def test_malformed_canonical_npz_is_rejected(
    tmp_path: Path,
    name: str,
    update: dict[str, object | None],
) -> None:
    path = tmp_path / f"{name}.npz"
    payload = _canonical_payload()
    for key, value in update.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    np.savez(path, **payload)

    with pytest.raises(SchemaValidationError):
        load_field(path)


def test_nonfinite_complex_field_is_rejected_in_strict_mode(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.npz"
    field = np.ones((3, 5), dtype=np.complex128)
    field[1, 2] = complex(float("nan"), 0)
    np.savez(path, **_canonical_payload(field=field))

    with pytest.raises(SchemaValidationError):
        load_field(path, strict=True)


def test_unknown_extension_and_missing_file_have_structured_errors(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "field.bin"
    unknown.write_bytes(b"not an optical field")

    with pytest.raises(UnsupportedFormatError):
        inspect_field(unknown)
    with pytest.raises(FileNotFoundError):
        inspect_field(tmp_path / "missing.npz")
