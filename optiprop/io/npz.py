"""Safe canonical and explicitly-mapped NPZ field I/O."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile

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
    SchemaValidationError,
    UnsafeDatasetError,
)
from .mapping import field_from_mapping
from .models import (
    ArrayInspection,
    DatasetInspection,
    ExportOptions,
    ExportReport,
    FieldFormat,
    ImportMapping,
    ImportResult,
    MappingConfidence,
)


SCHEMA_NAME = "optiprop.field2d"
SCHEMA_VERSION = 1
AXIS_ORDER = "C,Y,X"
COORDINATE_CONVENTION = "sample_center_increasing_xy"
FIELD_REPRESENTATION = "relative_electric_field_complex_amplitude"
TIME_CONVENTION = "exp(-i*omega*t)"

_REQUIRED_KEYS = frozenset(
    {
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
        "axis_order",
        "coordinate_convention",
        "field_representation",
        "time_convention",
    }
)
_MAX_MEMBERS = 256
_MAX_MEMBER_BYTES = 4 * 1024**3
_MAX_TOTAL_BYTES = 8 * 1024**3
_MAX_COMPRESSION_RATIO = 200
_COMPRESSION_RATIO_MIN_BYTES = 1024 * 1024


def inspect_npz(path: str | os.PathLike[str]) -> DatasetInspection:
    """Inspect an NPZ without enabling pickle or inferring physical units."""

    source = _existing_file(path)
    _preflight_zip(source)
    arrays: list[ArrayInspection] = []
    scalar_metadata: dict[str, Any] = {}
    issues: list[ValidationIssue] = []
    unsafe_object_keys: list[str] = []
    payload: dict[str, np.ndarray] = {}

    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SchemaValidationError(f"Cannot open NPZ dataset: {exc}") from exc
    with archive:
        keys = tuple(archive.files)
        for key in keys:
            try:
                value = archive[key]
            except ValueError as exc:
                if "Object arrays cannot be loaded" not in str(exc):
                    raise SchemaValidationError(
                        f"Cannot read NPZ member {key!r}: {exc}"
                    ) from exc
                arrays.append(
                    ArrayInspection(
                        key=key,
                        shape=(),
                        dtype="object",
                        is_complex=False,
                        is_object=True,
                        is_finite=None,
                    )
                )
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "io.unsafe_object_array",
                        (
                            f"NPZ member {key!r} uses object dtype and cannot "
                            "be loaded without pickle."
                        ),
                        details={"key": key},
                    )
                )
                unsafe_object_keys.append(key)
                continue
            payload[key] = value
            arrays.append(_inspect_array(key, value))
            if value.ndim == 0 and value.dtype.kind in "biufcUS":
                scalar_metadata[key] = _safe_scalar(value)

    if unsafe_object_keys:
        raise UnsafeDatasetError(
            "NPZ contains object arrays that would require pickle loading: "
            + ", ".join(repr(key) for key in unsafe_object_keys)
            + "."
        )

    schema_name = _optional_scalar_text(payload.get("schema_name"))
    schema_version = _optional_scalar_int(payload.get("schema_version"))
    is_canonical = False
    confidence = MappingConfidence.NONE
    suggestion = None
    if schema_name == SCHEMA_NAME:
        try:
            _parse_canonical_payload(payload, strict=True)
        except (SchemaValidationError, UnsafeDatasetError) as exc:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "io.canonical_schema_invalid",
                    str(exc),
                )
            )
        else:
            is_canonical = True
            confidence = MappingConfidence.EXACT
    else:
        suggestion, confidence = _legacy_suggestion(tuple(payload))
        issues.append(
            _issue(
                Severity.WARNING,
                "io.noncanonical_requires_mapping",
                (
                    "This NPZ is not canonical; explicit keys, units, and "
                    "orientation are required before import."
                ),
            )
        )

    return DatasetInspection(
        path=source,
        format=FieldFormat.NPZ,
        keys=tuple(item.key for item in arrays),
        arrays=tuple(arrays),
        scalar_metadata=scalar_metadata,
        metadata_candidates=_metadata_candidates(tuple(payload)),
        schema_name=schema_name,
        schema_version=schema_version,
        is_canonical=is_canonical,
        mapping_confidence=confidence,
        suggested_mapping=suggestion,
        warnings=tuple(issues),
    )


def load_npz_with_report(
    path: str | os.PathLike[str],
    mapping: ImportMapping | None = None,
    *,
    strict: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ImportResult:
    """Load canonical NPZ or apply an explicit mapping to a legacy NPZ."""

    if not isinstance(strict, bool):
        raise TypeError("strict must be a bool.")
    if mapping is not None and not isinstance(mapping, ImportMapping):
        raise TypeError("mapping must be an ImportMapping or None.")
    if dtype is not None and dtype not in (torch.complex64, torch.complex128):
        raise TypeError("dtype must be torch.complex64 or torch.complex128.")
    inspection = inspect_npz(path)
    source = inspection.path
    with np.load(source, allow_pickle=False) as archive:
        try:
            payload = {key: archive[key] for key in archive.files}
        except ValueError as exc:
            if "Object arrays cannot be loaded" in str(exc):
                raise UnsafeDatasetError(
                    "NPZ contains object arrays; pickle loading is forbidden."
                ) from exc
            raise SchemaValidationError(f"Cannot read NPZ dataset: {exc}") from exc

    warnings = list(inspection.warnings)
    claimed_canonical = inspection.schema_name == SCHEMA_NAME
    if claimed_canonical:
        if mapping is not None:
            raise ValueError("Canonical NPZ import does not accept a mapping.")
        parsed, parse_warnings = _parse_canonical_payload(payload, strict=strict)
        warnings.extend(parse_warnings)
        field = _field_from_canonical(parsed)
    else:
        if mapping is None:
            raise AmbiguousMappingError(
                "Noncanonical NPZ requires an explicit ImportMapping; "
                "inspection suggestions never authorize unit/orientation guesses."
            )
        field = field_from_mapping(payload, mapping, source_path=source)

    if device is not None or dtype is not None:
        field = field.to(device=device, dtype=dtype)
    return ImportResult(
        field=field,
        inspection=inspection,
        mapping=mapping,
        warnings=tuple(
            issue
            for issue in warnings
            if issue.severity is not Severity.ERROR
        ),
    )


def load_npz(
    path: str | os.PathLike[str],
    mapping: ImportMapping | None = None,
    *,
    strict: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Field2D:
    return load_npz_with_report(
        path,
        mapping,
        strict=strict,
        device=device,
        dtype=dtype,
    ).field


def save_npz(
    field: Field2D,
    path: str | os.PathLike[str],
    *,
    options: ExportOptions | None = None,
) -> ExportReport:
    """Write schema-v1 canonical NPZ using only primitive NumPy arrays."""

    if not isinstance(field, Field2D):
        raise TypeError("field must be a Field2D.")
    options = ExportOptions() if options is None else options
    if not isinstance(options, ExportOptions):
        raise TypeError("options must be an ExportOptions or None.")
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("Canonical NPZ export path must end in '.npz'.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Export directory does not exist: {destination.parent}"
        )
    if not torch.all(torch.isfinite(field.data.real)).item() or not torch.all(
        torch.isfinite(field.data.imag)
    ).item():
        raise SchemaValidationError("Cannot export a field containing NaN/Inf.")

    data = (
        field.data.detach()
        .to(device="cpu")
        .contiguous()
        .numpy()
        .copy()
    )
    metadata = dict(field.metadata) if options.include_metadata else {}
    metadata_json = json.dumps(
        _json_encode(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload: dict[str, np.ndarray] = {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "field": data,
        "components": np.asarray(field.components, dtype=np.str_),
        "dx_m": np.asarray(field.grid.dx, dtype=np.float64),
        "dy_m": np.asarray(field.grid.dy, dtype=np.float64),
        "wavelength_m": np.asarray(field.wavelength_m, dtype=np.float64),
        "medium_index_real": np.asarray(field.medium_index.real, dtype=np.float64),
        "medium_index_imag": np.asarray(field.medium_index.imag, dtype=np.float64),
        "x_center_m": np.asarray(field.grid.x_center, dtype=np.float64),
        "y_center_m": np.asarray(field.grid.y_center, dtype=np.float64),
        "z_m": np.asarray(field.z_m, dtype=np.float64),
        "metadata_json": np.asarray(metadata_json),
        "axis_order": np.asarray(AXIS_ORDER),
        "coordinate_convention": np.asarray(COORDINATE_CONVENTION),
        "field_representation": np.asarray(FIELD_REPRESENTATION),
        "time_convention": np.asarray(TIME_CONVENTION),
    }
    if any(value.dtype.hasobject for value in payload.values()):
        raise UnsafeDatasetError("Canonical payload unexpectedly contains object dtype.")

    actual_compressed = options.compressed
    export_warnings: list[ValidationIssue] = []
    if options.atomic:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                _write_archive(handle, payload, options.compressed)
                handle.flush()
                os.fsync(handle.fileno())
            actual_compressed = _rewrite_if_overcompressed(
                temporary_path,
                payload,
                requested_compressed=options.compressed,
                warnings=export_warnings,
            )
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    else:
        with destination.open("w+b") as handle:
            _write_archive(handle, payload, options.compressed)
            handle.flush()
            os.fsync(handle.fileno())
        actual_compressed = _rewrite_if_overcompressed(
            destination,
            payload,
            requested_compressed=options.compressed,
            warnings=export_warnings,
        )

    return ExportReport(
        path=destination,
        format=FieldFormat.NPZ,
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        bytes_written=destination.stat().st_size,
        sha256=_file_sha256(destination),
        atomic=options.atomic,
        compressed=actual_compressed,
        warnings=tuple(export_warnings),
    )


def _write_archive(
    handle,
    payload: Mapping[str, np.ndarray],
    compressed: bool,
) -> None:
    writer = np.savez_compressed if compressed else np.savez
    # np.savez has no allow_pickle argument.  Safety comes from validating
    # every payload dtype above and from reader-side allow_pickle=False.
    writer(handle, **payload)


def _rewrite_if_overcompressed(
    path: Path,
    payload: Mapping[str, np.ndarray],
    *,
    requested_compressed: bool,
    warnings: list[ValidationIssue],
) -> bool:
    """Ensure the writer never emits an archive rejected by its own reader."""

    try:
        _preflight_zip(path)
        return requested_compressed
    except UnsafeDatasetError as exc:
        if not requested_compressed or "compression-ratio" not in str(exc):
            raise
    with path.open("w+b") as handle:
        _write_archive(handle, payload, compressed=False)
        handle.flush()
        os.fsync(handle.fileno())
    _preflight_zip(path)
    warnings.append(
        _issue(
            Severity.WARNING,
            "io.compression_fallback",
            (
                "Compressed output exceeded the reader safety ratio and was "
                "rewritten uncompressed to preserve safe round-trip loading."
            ),
        )
    )
    return False


def _parse_canonical_payload(
    payload: Mapping[str, np.ndarray],
    *,
    strict: bool,
) -> tuple[dict[str, Any], tuple[ValidationIssue, ...]]:
    missing = sorted(_REQUIRED_KEYS - payload.keys())
    if missing:
        raise SchemaValidationError(
            f"Canonical NPZ is missing required keys: {', '.join(missing)}."
        )
    unknown = sorted(payload.keys() - _REQUIRED_KEYS)
    warnings = []
    if unknown:
        message = f"Canonical NPZ contains unknown keys: {', '.join(unknown)}."
        if strict:
            raise SchemaValidationError(message)
        warnings.append(
            _issue(Severity.WARNING, "io.unknown_canonical_keys", message)
        )
    if any(value.dtype.hasobject for value in payload.values()):
        raise UnsafeDatasetError(
            "Canonical NPZ contains object dtype; pickle loading is forbidden."
        )
    if _scalar_text(payload["schema_name"], "schema_name") != SCHEMA_NAME:
        raise SchemaValidationError("schema_name is not 'optiprop.field2d'.")
    if _scalar_int(payload["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"Unsupported schema_version; expected {SCHEMA_VERSION}."
        )
    conventions = {
        "axis_order": AXIS_ORDER,
        "coordinate_convention": COORDINATE_CONVENTION,
        "field_representation": FIELD_REPRESENTATION,
        "time_convention": TIME_CONVENTION,
    }
    for key, expected in conventions.items():
        actual = _scalar_text(payload[key], key)
        if actual != expected:
            raise SchemaValidationError(
                f"{key} must be {expected!r}; got {actual!r}."
            )

    data = np.asarray(payload["field"])
    if data.dtype not in (np.dtype("complex64"), np.dtype("complex128")):
        raise SchemaValidationError("field must be complex64 or complex128.")
    if data.ndim == 2 and not strict:
        data = data[np.newaxis, ...]
        warnings.append(
            _issue(
                Severity.WARNING,
                "io.legacy_scalar_shape",
                "Accepted legacy canonical scalar [ny,nx] as [1,ny,nx].",
            )
        )
    if data.ndim != 3 or data.shape[0] not in (1, 2):
        raise SchemaValidationError(
            "field must have canonical shape [1|2, ny, nx]."
        )
    if data.shape[-2] <= 0 or data.shape[-1] <= 0:
        raise SchemaValidationError("field spatial dimensions must be positive.")
    if not np.isfinite(data.real).all() or not np.isfinite(data.imag).all():
        raise SchemaValidationError("field contains NaN or infinity.")
    components_array = np.asarray(payload["components"])
    if components_array.ndim != 1 or components_array.dtype.kind not in "US":
        raise SchemaValidationError("components must be a 1-D Unicode/string array.")
    components = tuple(str(value) for value in components_array.tolist())
    expected_components = (
        SCALAR_COMPONENTS if data.shape[0] == 1 else EX_EY_COMPONENTS
    )
    if components != expected_components:
        raise SchemaValidationError(
            f"components must be {expected_components!r} for field shape."
        )

    dx = _positive_scalar(payload["dx_m"], "dx_m")
    dy = _positive_scalar(payload["dy_m"], "dy_m")
    wavelength = _positive_scalar(payload["wavelength_m"], "wavelength_m")
    medium_real = _finite_scalar(
        payload["medium_index_real"], "medium_index_real"
    )
    medium_imag = _finite_scalar(
        payload["medium_index_imag"], "medium_index_imag"
    )
    if medium_real <= 0:
        raise SchemaValidationError("medium_index_real must be positive.")
    x_center = _finite_scalar(payload["x_center_m"], "x_center_m")
    y_center = _finite_scalar(payload["y_center_m"], "y_center_m")
    z_m = _finite_scalar(payload["z_m"], "z_m")
    metadata_text = _scalar_text(payload["metadata_json"], "metadata_json")
    try:
        metadata = json.loads(
            metadata_text,
            object_hook=_json_decode_hook,
            parse_constant=lambda value: (_raise_json_constant(value)),
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise SchemaValidationError(f"metadata_json is invalid: {exc}") from exc
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) for key in metadata
    ):
        raise SchemaValidationError("metadata_json root must be a JSON object.")
    return (
        {
            "data": np.array(data, copy=True),
            "components": components,
            "dx": dx,
            "dy": dy,
            "wavelength": wavelength,
            "medium_index": complex(medium_real, medium_imag),
            "x_center": x_center,
            "y_center": y_center,
            "z_m": z_m,
            "metadata": metadata,
        },
        tuple(warnings),
    )


def _field_from_canonical(parsed: Mapping[str, Any]) -> Field2D:
    data = parsed["data"]
    native_dtype = np.complex64 if data.dtype.itemsize == 8 else np.complex128
    tensor = torch.from_numpy(data.astype(native_dtype, copy=True))
    grid = Grid2D(
        nx=tensor.shape[-1],
        ny=tensor.shape[-2],
        dx=parsed["dx"],
        dy=parsed["dy"],
        x_center=parsed["x_center"],
        y_center=parsed["y_center"],
    )
    return Field2D(
        data=tensor,
        grid=grid,
        wavelength_m=parsed["wavelength"],
        medium_index=parsed["medium_index"],
        components=parsed["components"],
        z_m=parsed["z_m"],
        metadata=parsed["metadata"],
    )


def _inspect_array(key: str, value: np.ndarray) -> ArrayInspection:
    is_complex = np.iscomplexobj(value)
    numeric = value.dtype.kind in "biufc"
    finite = bool(np.isfinite(value).all()) if numeric else None
    value_min = value_max = magnitude_min = magnitude_max = None
    if value.size and numeric:
        if is_complex:
            magnitude = np.abs(value)
            magnitude_min = float(np.min(magnitude))
            magnitude_max = float(np.max(magnitude))
        else:
            value_min = float(np.min(value))
            value_max = float(np.max(value))
    return ArrayInspection(
        key=key,
        shape=tuple(int(size) for size in value.shape),
        dtype=str(value.dtype),
        is_complex=is_complex,
        is_object=value.dtype.hasobject,
        is_finite=finite,
        value_min=value_min,
        value_max=value_max,
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
    )


def _legacy_suggestion(
    keys: tuple[str, ...],
) -> tuple[ImportMapping | None, MappingConfidence]:
    key_set = set(keys)
    pairs = [
        ("U_after_Ex", "U_after_Ey"),
        ("EX", "EY"),
        ("Ex", "Ey"),
    ]
    pair_matches = [pair for pair in pairs if set(pair) <= key_set]
    singles = [key for key in ("field", "U") if key in key_set]
    if len(pair_matches) + len(singles) != 1:
        return None, (
            MappingConfidence.AMBIGUOUS
            if pair_matches or singles
            else MappingConfidence.NONE
        )
    if pair_matches:
        return (
            ImportMapping(
                data_keys=pair_matches[0],
                components=EX_EY_COMPONENTS,
            ),
            MappingConfidence.SUGGESTED,
        )
    return (
        ImportMapping(data_keys=(singles[0],), components=SCALAR_COMPONENTS),
        MappingConfidence.SUGGESTED,
    )


def _metadata_candidates(keys: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    aliases = {
        "dx": ("dx_m", "dx", "pixel_size"),
        "dy": ("dy_m", "dy", "pixel_size"),
        "wavelength": (
            "wavelength_m",
            "wavelength",
            "lambda",
            "design_lambda",
        ),
        "nx": ("Nx", "nx"),
        "ny": ("Ny", "ny"),
    }
    key_set = set(keys)
    return {
        category: tuple(alias for alias in candidates if alias in key_set)
        for category, candidates in aliases.items()
        if any(alias in key_set for alias in candidates)
    }


def _json_encode(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata floats must be finite.")
        return value
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("metadata complex values must be finite.")
        return {
            "__optiprop_type__": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings.")
        return {key: _json_encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_encode(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_encode(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False),
        )
    if isinstance(value, Path):
        return {"__optiprop_type__": "path", "value": str(value)}
    if isinstance(value, (bytes, bytearray)):
        return {
            "__optiprop_type__": "bytes",
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _json_encode(value.item())
    if isinstance(value, (torch.dtype, torch.device)):
        return {
            "__optiprop_type__": type(value).__name__,
            "value": str(value),
        }
    if isinstance(value, torch.Tensor):
        raise TypeError("Tensor metadata is not supported; save it as an asset.")
    raise TypeError(
        f"Metadata value of type {type(value).__name__} is not JSON serializable."
    )


def _json_decode_hook(value: dict[str, Any]) -> Any:
    tag = value.get("__optiprop_type__")
    if tag == "complex" and set(value) == {
        "__optiprop_type__",
        "real",
        "imag",
    }:
        return complex(value["real"], value["imag"])
    if tag == "path" and set(value) == {"__optiprop_type__", "value"}:
        return value["value"]
    if tag == "bytes" and set(value) == {"__optiprop_type__", "base64"}:
        try:
            return base64.b64decode(value["base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid base64 metadata value.") from exc
    if tag in ("dtype", "device") and set(value) == {
        "__optiprop_type__",
        "value",
    }:
        return value["value"]
    return value


def _preflight_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise SchemaValidationError("NPZ is not a valid ZIP archive.") from exc
    if len(members) > _MAX_MEMBERS:
        raise UnsafeDatasetError("NPZ contains too many members.")
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise UnsafeDatasetError("NPZ contains duplicate ZIP member names.")
    total = 0
    for member in members:
        name = member.filename
        if (
            not name.endswith(".npy")
            or "/" in name
            or "\\" in name
            or name.startswith(".")
        ):
            raise UnsafeDatasetError(f"Unsafe NPZ member name: {name!r}.")
        if member.file_size > _MAX_MEMBER_BYTES:
            raise UnsafeDatasetError(f"NPZ member {name!r} is too large.")
        total += member.file_size
        if member.compress_size == 0:
            if member.file_size:
                raise UnsafeDatasetError(
                    f"NPZ member {name!r} has an invalid compressed size."
                )
        elif (
            member.file_size >= _COMPRESSION_RATIO_MIN_BYTES
            and member.file_size / member.compress_size > _MAX_COMPRESSION_RATIO
        ):
            raise UnsafeDatasetError(
                f"NPZ member {name!r} exceeds the compression-ratio limit."
            )
    if total > _MAX_TOTAL_BYTES:
        raise UnsafeDatasetError("NPZ uncompressed payload is too large.")


def _existing_file(path: str | os.PathLike[str]) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Field dataset does not exist: {source}")
    if source.suffix.lower() != ".npz":
        raise ValueError("NPZ path must end in '.npz'.")
    return source


def _safe_scalar(value: np.ndarray) -> Any:
    item = value.item()
    if isinstance(item, np.generic):
        item = item.item()
    return item


def _scalar_text(value: np.ndarray, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise SchemaValidationError(f"{name} must be a scalar string.")
    return str(value.item())


def _scalar_int(value: np.ndarray, name: str) -> int:
    if value.ndim != 0 or value.dtype.kind not in "iu":
        raise SchemaValidationError(f"{name} must be a scalar integer.")
    return int(value.item())


def _finite_scalar(value: np.ndarray, name: str) -> float:
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise SchemaValidationError(f"{name} must be a scalar real number.")
    normalized = float(value.item())
    if not math.isfinite(normalized):
        raise SchemaValidationError(f"{name} must be finite.")
    return normalized


def _positive_scalar(value: np.ndarray, name: str) -> float:
    normalized = _finite_scalar(value, name)
    if normalized <= 0:
        raise SchemaValidationError(f"{name} must be positive.")
    return normalized


def _optional_scalar_text(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    try:
        return _scalar_text(value, "schema_name")
    except SchemaValidationError:
        return None


def _optional_scalar_int(value: np.ndarray | None) -> int | None:
    if value is None:
        return None
    try:
        return _scalar_int(value, "schema_version")
    except SchemaValidationError:
        return None


def _raise_json_constant(value: str):
    raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")


def _issue(
    severity: Severity,
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        details={} if details is None else details,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AXIS_ORDER",
    "COORDINATE_CONVENTION",
    "FIELD_REPRESENTATION",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TIME_CONVENTION",
    "inspect_npz",
    "load_npz",
    "load_npz_with_report",
    "save_npz",
]
