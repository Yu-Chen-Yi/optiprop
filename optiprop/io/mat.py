"""Canonical and explicitly mapped MATLAB field I/O."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import numpy as np
import torch

from ..core import (
    EX_EY_COMPONENTS,
    SCALAR_COMPONENTS,
    Field2D,
    Grid2D,
    Severity,
    ValidationIssue,
)
from .errors import (
    AmbiguousMappingError,
    FieldIOError,
    SchemaValidationError,
    UnsafeDatasetError,
    UnsupportedFormatError,
)
from .mapping import field_from_mapping
from .models import (
    ArrayInspection,
    DatasetInspection,
    ExportOptions,
    ExportReport,
    FieldFormat,
    ImportMapping,
    MappingConfidence,
)


SCHEMA_NAME = "optiprop.field2d"
SCHEMA_VERSION = 1
_REQUIRED_CANONICAL_KEYS = {
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
}
_SAFE_MAT_CLASSES = {
    "char",
    "logical",
    "double",
    "single",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
}


def inspect_mat(path: Path | str) -> DatasetInspection:
    """Inspect a MAT v4/v5-v7.2 file without interpreting legacy units."""

    path = _existing_path(path)
    _reject_v73(path)
    scipy_io = _scipy_io()
    try:
        inventory = scipy_io.whosmat(path)
    except Exception as exc:
        raise FieldIOError(f"Could not inspect MAT file {path}: {exc}") from exc

    safe_names = [
        name for name, _shape, storage_class in inventory
        if storage_class in _SAFE_MAT_CLASSES
    ]
    try:
        payload = scipy_io.loadmat(
            path,
            variable_names=safe_names,
            squeeze_me=False,
            struct_as_record=True,
            verify_compressed_data_integrity=True,
        )
    except Exception as exc:
        raise FieldIOError(f"Could not read MAT inventory {path}: {exc}") from exc
    payload = {
        key: value for key, value in payload.items() if not key.startswith("__")
    }

    arrays = []
    scalar_metadata: dict[str, Any] = {}
    storage_by_name = {
        name: storage_class for name, _shape, storage_class in inventory
    }
    shape_by_name = {name: tuple(shape) for name, shape, _class in inventory}
    for name, _shape, _class in inventory:
        if name in payload:
            array = np.asarray(payload[name])
            inspection = _array_inspection(name, array)
            if array.size == 1 and not inspection.is_object:
                scalar_metadata[name] = _python_scalar(array)
        else:
            inspection = ArrayInspection(
                key=name,
                shape=shape_by_name[name],
                dtype=f"matlab:{storage_by_name[name]}",
                is_complex=False,
                is_object=True,
                is_finite=None,
            )
        arrays.append(inspection)

    schema_name = _optional_string(payload.get("schema_name"))
    schema_version = _optional_integer(payload.get("schema_version"))
    is_canonical = schema_name == SCHEMA_NAME
    suggestion = None if is_canonical else _suggest_mapping(tuple(shape_by_name))
    warnings = () if is_canonical else (
        ValidationIssue(
            severity=Severity.WARNING,
            code="mapping.required",
            message=(
                "This MAT file is noncanonical. Confirm key mapping, units, "
                "and axis orientation before loading."
            ),
        ),
    )
    return DatasetInspection(
        path=path,
        format=FieldFormat.MAT,
        keys=tuple(name for name, _shape, _class in inventory),
        arrays=tuple(arrays),
        scalar_metadata=scalar_metadata,
        metadata_candidates=_metadata_candidates(tuple(shape_by_name)),
        schema_name=schema_name,
        schema_version=schema_version,
        is_canonical=is_canonical,
        mapping_confidence=(
            MappingConfidence.EXACT
            if is_canonical
            else (
                MappingConfidence.AMBIGUOUS
                if suggestion is not None
                else MappingConfidence.NONE
            )
        ),
        suggested_mapping=suggestion,
        warnings=warnings,
    )


def load_mat(
    path: Path | str,
    mapping: ImportMapping | None = None,
    *,
    strict: bool = True,
) -> Field2D:
    """Load canonical MAT or apply an explicit noncanonical mapping."""

    path = _existing_path(path)
    inspection = inspect_mat(path)
    if not inspection.is_canonical and mapping is None:
        raise AmbiguousMappingError(
            "Noncanonical MAT input requires an explicit ImportMapping."
        )
    payload = _load_safe_payload(path)
    if inspection.is_canonical:
        if mapping is not None:
            raise ValueError("Canonical MAT input does not accept ImportMapping.")
        return _field_from_canonical(payload, strict=strict)
    return field_from_mapping(payload, mapping, source_path=path)


def save_mat(
    field: Field2D,
    path: Path | str,
    *,
    options: ExportOptions | None = None,
) -> ExportReport:
    """Atomically save canonical MAT v5 using SciPy."""

    if not isinstance(field, Field2D):
        raise TypeError("field must be a Field2D.")
    options = ExportOptions() if options is None else options
    if not isinstance(options, ExportOptions):
        raise TypeError("options must be an ExportOptions or None.")
    path = Path(path).resolve()
    if path.suffix.lower() != ".mat":
        raise UnsupportedFormatError("MAT output path must use the .mat extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_mat_payload(field, include_metadata=options.include_metadata)
    scipy_io = _scipy_io()

    if options.atomic:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            scipy_io.savemat(
                temporary_path,
                payload,
                appendmat=False,
                do_compression=options.compressed,
                long_field_names=True,
                oned_as="row",
            )
            with temporary_path.open("r+b") as temporary:
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    else:
        scipy_io.savemat(
            path,
            payload,
            appendmat=False,
            do_compression=options.compressed,
            long_field_names=True,
            oned_as="row",
        )
        with path.open("r+b") as output:
            output.flush()
            os.fsync(output.fileno())
    encoded = path.read_bytes()
    return ExportReport(
        path=path,
        format=FieldFormat.MAT,
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        bytes_written=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        atomic=options.atomic,
        compressed=options.compressed,
    )


def canonical_mat_payload(
    field: Field2D,
    *,
    include_metadata: bool = True,
) -> dict[str, np.ndarray]:
    """Return primitive canonical variables suitable for ``savemat``."""

    data = (
        field.data.detach()
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .cpu()
        .numpy()
    )
    if not np.isfinite(data.real).all() or not np.isfinite(data.imag).all():
        raise SchemaValidationError("Cannot export a non-finite field.")
    metadata = _metadata_json(field.metadata if include_metadata else {})
    return {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "field": data,
        "components": np.asarray(field.components, dtype="<U8"),
        "axis_order": np.asarray("C,Y,X"),
        "coordinate_convention": np.asarray("sample_center_increasing_xy"),
        "field_representation": np.asarray(
            "relative_electric_field_complex_amplitude"
        ),
        "time_convention": np.asarray("exp(-i*omega*t)"),
        "dx_m": np.asarray(field.grid.dx, dtype=np.float64),
        "dy_m": np.asarray(field.grid.dy, dtype=np.float64),
        "wavelength_m": np.asarray(field.wavelength_m, dtype=np.float64),
        "medium_index_real": np.asarray(field.medium_index.real, dtype=np.float64),
        "medium_index_imag": np.asarray(field.medium_index.imag, dtype=np.float64),
        "x_center_m": np.asarray(field.grid.x_center, dtype=np.float64),
        "y_center_m": np.asarray(field.grid.y_center, dtype=np.float64),
        "z_m": np.asarray(field.z_m, dtype=np.float64),
        "metadata_json": np.asarray(metadata),
    }


def _field_from_canonical(
    payload: Mapping[str, Any],
    *,
    strict: bool,
) -> Field2D:
    missing = _REQUIRED_CANONICAL_KEYS.difference(payload)
    if missing:
        raise SchemaValidationError(
            f"Canonical MAT is missing required keys: {sorted(missing)!r}."
        )
    if _string(payload["schema_name"], "schema_name") != SCHEMA_NAME:
        raise SchemaValidationError("Unsupported canonical schema_name.")
    if _integer(payload["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise SchemaValidationError("Unsupported canonical schema_version.")
    for key, expected in (
        ("axis_order", "C,Y,X"),
        ("coordinate_convention", "sample_center_increasing_xy"),
        ("field_representation", "relative_electric_field_complex_amplitude"),
        ("time_convention", "exp(-i*omega*t)"),
    ):
        if key in payload and _string(payload[key], key) != expected:
            raise SchemaValidationError(f"Unsupported canonical {key}.")

    data = np.asarray(payload["field"])
    if data.dtype not in (np.dtype("complex64"), np.dtype("complex128")):
        raise SchemaValidationError(
            "Canonical field must use complex64 or complex128."
        )
    components = _components(payload["components"])
    if data.ndim == 2 and components == SCALAR_COMPONENTS:
        data = data[None]
    if data.ndim != 3 or data.shape[0] != len(components):
        raise SchemaValidationError(
            "Canonical field shape must be [C,ny,nx] and match components."
        )
    if strict and (
        not np.isfinite(data.real).all() or not np.isfinite(data.imag).all()
    ):
        raise SchemaValidationError("Canonical field contains NaN or infinity.")

    dx = _positive_scalar(payload["dx_m"], "dx_m")
    dy = _positive_scalar(payload["dy_m"], "dy_m")
    wavelength = _positive_scalar(payload["wavelength_m"], "wavelength_m")
    medium_index = complex(
        _finite_scalar(payload["medium_index_real"], "medium_index_real"),
        _finite_scalar(payload["medium_index_imag"], "medium_index_imag"),
    )
    if medium_index.real <= 0:
        raise SchemaValidationError("medium_index_real must be positive.")
    x_center = _finite_scalar(payload["x_center_m"], "x_center_m")
    y_center = _finite_scalar(payload["y_center_m"], "y_center_m")
    z_m = _finite_scalar(payload["z_m"], "z_m")
    metadata = _decode_metadata(payload["metadata_json"])
    try:
        return Field2D(
            data=torch.from_numpy(np.ascontiguousarray(data)),
            grid=Grid2D(
                nx=data.shape[-1],
                ny=data.shape[-2],
                dx=dx,
                dy=dy,
                x_center=x_center,
                y_center=y_center,
            ),
            wavelength_m=wavelength,
            medium_index=medium_index,
            components=components,
            z_m=z_m,
            metadata=metadata,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"Invalid canonical MAT field: {exc}") from exc


def _load_safe_payload(path: Path) -> dict[str, np.ndarray]:
    scipy_io = _scipy_io()
    try:
        inventory = scipy_io.whosmat(path)
    except Exception as exc:
        raise FieldIOError(f"Could not inspect MAT file {path}: {exc}") from exc
    unsafe = [
        (name, storage_class)
        for name, _shape, storage_class in inventory
        if storage_class not in _SAFE_MAT_CLASSES
    ]
    if unsafe:
        raise UnsafeDatasetError(
            f"MAT contains unsupported cell/struct/object variables: {unsafe!r}."
        )
    try:
        payload = scipy_io.loadmat(
            path,
            variable_names=[name for name, _shape, _class in inventory],
            squeeze_me=False,
            struct_as_record=True,
            verify_compressed_data_integrity=True,
        )
    except Exception as exc:
        raise FieldIOError(f"Could not load MAT file {path}: {exc}") from exc
    return {
        key: np.asarray(value)
        for key, value in payload.items()
        if not key.startswith("__")
    }


def _suggest_mapping(keys: tuple[str, ...]) -> ImportMapping | None:
    key_set = set(keys)
    pairs = (
        ("EX", "EY"),
        ("Ex", "Ey"),
        ("U_after_Ex", "U_after_Ey"),
    )
    selected = next((pair for pair in pairs if set(pair) <= key_set), None)
    components = EX_EY_COMPONENTS
    if selected is None:
        single = next((key for key in ("U", "field") if key in key_set), None)
        if single is None:
            return None
        selected = (single,)
        components = SCALAR_COMPONENTS
    pixel = next(
        (key for key in ("pixel_size",) if key in key_set),
        None,
    )
    dx = next((key for key in ("dx", "dx_m") if key in key_set), None)
    dy = next((key for key in ("dy", "dy_m") if key in key_set), None)
    wavelength = next(
        (
            key
            for key in ("wavelength", "lambda", "design_lambda", "wavelength_m")
            if key in key_set
        ),
        None,
    )
    nx = next((key for key in ("nx", "Nx") if key in key_set), None)
    ny = next((key for key in ("ny", "Ny") if key in key_set), None)
    return ImportMapping(
        data_keys=selected,
        components=components,
        pixel_size_key=pixel,
        dx_key=None if pixel else dx,
        dy_key=None if pixel else dy,
        wavelength_key=wavelength,
        nx_key=nx,
        ny_key=ny,
    )


def _metadata_candidates(keys: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    aliases = {
        "field": (
            "field",
            "U",
            "EX",
            "EY",
            "Ex",
            "Ey",
            "U_after_Ex",
            "U_after_Ey",
        ),
        "spacing": ("dx_m", "dy_m", "dx", "dy", "pixel_size"),
        "wavelength": (
            "wavelength_m",
            "wavelength",
            "lambda",
            "design_lambda",
        ),
    }
    key_set = set(keys)
    return {
        name: tuple(key for key in candidates if key in key_set)
        for name, candidates in aliases.items()
    }


def _array_inspection(key: str, array: np.ndarray) -> ArrayInspection:
    is_object = array.dtype.hasobject or array.dtype.fields is not None
    numeric = array.dtype.kind in "biufc" and not is_object
    finite = bool(np.isfinite(array).all()) if numeric else None
    value_min = value_max = magnitude_min = magnitude_max = None
    if numeric and array.size:
        if np.iscomplexobj(array):
            magnitude = np.abs(array)
            magnitude_min = float(np.nanmin(magnitude))
            magnitude_max = float(np.nanmax(magnitude))
        else:
            value_min = float(np.nanmin(array))
            value_max = float(np.nanmax(array))
    return ArrayInspection(
        key=key,
        shape=tuple(int(value) for value in array.shape),
        dtype=str(array.dtype),
        is_complex=bool(np.iscomplexobj(array)),
        is_object=is_object,
        is_finite=finite,
        value_min=value_min,
        value_max=value_max,
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
    )


def _components(value: Any) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.dtype.kind not in "SU":
        raise SchemaValidationError("components must be a Unicode/string array.")
    if array.ndim == 2 and array.dtype.itemsize // np.dtype("U1").itemsize == 1:
        values = tuple("".join(row.tolist()).strip() for row in array)
    else:
        values = tuple(str(item).strip() for item in array.reshape(-1))
    values = tuple(item for item in values if item)
    if values not in (SCALAR_COMPONENTS, EX_EY_COMPONENTS):
        raise SchemaValidationError(
            "components must be ['scalar'] or ['Ex','Ey']."
        )
    return values


def _decode_metadata(value: Any) -> dict[str, Any]:
    text = _string(value, "metadata_json")
    try:
        metadata = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SchemaValidationError("metadata_json is not strict JSON.") from exc
    if not isinstance(metadata, dict):
        raise SchemaValidationError("metadata_json root must be an object.")
    return metadata


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    if not isinstance(metadata, Mapping):
        raise TypeError("field metadata must be a mapping.")
    try:
        return json.dumps(
            _json_value(metadata),
            sort_keys=True,
            separators=(",", ":"),
            # MATLAB v5 char arrays are not a reliable UTF-8 container across
            # SciPy/MATLAB versions. JSON escapes keep the payload ASCII while
            # round-tripping full Unicode through json.loads().
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FieldIOError(f"Field metadata is not safely JSON serializable: {exc}") from exc


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata floats must be finite.")
        return value
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("metadata complex values must be finite.")
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (Enum, UUID, Path, torch.dtype, torch.device)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported metadata value {type(value).__name__}.")


def _string(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.dtype.kind not in "SU" or array.size == 0:
        raise SchemaValidationError(f"{label} must be a string scalar.")
    if array.ndim >= 2 and array.dtype.itemsize // np.dtype("U1").itemsize == 1:
        return "".join(str(item) for item in array.reshape(-1)).strip()
    if array.size != 1:
        raise SchemaValidationError(f"{label} must be a string scalar.")
    return str(array.reshape(-1)[0]).strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return _string(value, "string")
    except SchemaValidationError:
        return None


def _integer(value: Any, label: str) -> int:
    scalar = _finite_scalar(value, label)
    if not scalar.is_integer():
        raise SchemaValidationError(f"{label} must be an integer.")
    return int(scalar)


def _optional_integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return _integer(value, "integer")
    except SchemaValidationError:
        return None


def _finite_scalar(value: Any, label: str) -> float:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "biuf":
        raise SchemaValidationError(f"{label} must be one real numeric scalar.")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise SchemaValidationError(f"{label} must be finite.")
    return result


def _positive_scalar(value: Any, label: str) -> float:
    result = _finite_scalar(value, label)
    if result <= 0:
        raise SchemaValidationError(f"{label} must be positive.")
    return result


def _python_scalar(value: np.ndarray) -> Any:
    scalar = value.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _existing_path(path: Path | str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".mat":
        raise UnsupportedFormatError("MAT input path must use the .mat extension.")
    return path


def _reject_v73(path: Path) -> None:
    with path.open("rb") as stream:
        signature = stream.read(8)
    if signature == b"\x89HDF\r\n\x1a\n":
        raise UnsupportedFormatError(
            "MAT v7.3/HDF5 requires the optional h5py reader, which is not "
            "implemented in WP-08."
        )


def _scipy_io():
    try:
        from scipy import io as scipy_io
    except ImportError as exc:
        raise UnsupportedFormatError(
            "MAT support requires scipy>=1.7."
        ) from exc
    return scipy_io


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "canonical_mat_payload",
    "inspect_mat",
    "load_mat",
    "save_mat",
]
