"""Format-dispatching public field-I/O API."""

from __future__ import annotations

import os
from pathlib import Path

from ..core import Field2D
from .errors import UnsupportedFormatError
from .mat import inspect_mat, load_mat, save_mat
from .models import (
    DatasetInspection,
    ExportOptions,
    ExportReport,
    FieldFormat,
    ImportMapping,
)
from .npz import inspect_npz, load_npz, save_npz
from .zbf import inspect_zbf, load_zbf, save_zbf


def inspect_field(path: str | os.PathLike[str]) -> DatasetInspection:
    """Inspect a supported field container without guessing legacy semantics."""

    resolved = Path(path).expanduser().resolve()
    format_value = _format_from_suffix(resolved)
    if format_value is FieldFormat.NPZ:
        return inspect_npz(resolved)
    if format_value is FieldFormat.MAT:
        return inspect_mat(resolved)
    if format_value is FieldFormat.ZBF:
        return inspect_zbf(resolved)
    raise UnsupportedFormatError(f"Unsupported field format: {format_value.value}.")


def load_field(
    path: str | os.PathLike[str],
    mapping: ImportMapping | None = None,
    *,
    strict: bool = True,
) -> Field2D:
    """Load canonical field data or apply an explicit legacy mapping."""

    resolved = Path(path).expanduser().resolve()
    format_value = _format_from_suffix(resolved)
    if format_value is FieldFormat.NPZ:
        return load_npz(resolved, mapping, strict=strict)
    if format_value is FieldFormat.MAT:
        return load_mat(resolved, mapping, strict=strict)
    if format_value is FieldFormat.ZBF:
        return load_zbf(resolved, mapping, strict=strict)
    raise UnsupportedFormatError(f"Unsupported field format: {format_value.value}.")


def save_field(
    field: Field2D,
    path: str | os.PathLike[str],
    *,
    format: FieldFormat | str | None = None,
    options: ExportOptions | None = None,
) -> ExportReport:
    """Save a canonical field using a suffix-matched NPZ or MAT codec."""

    resolved = Path(path).expanduser().resolve()
    suffix_format = _format_from_suffix(resolved)
    explicit = suffix_format if format is None else _coerce_format(format)
    if explicit is not suffix_format:
        raise UnsupportedFormatError(
            f"Explicit format {explicit.value!r} does not match "
            f"destination extension {resolved.suffix!r}."
        )
    if explicit is FieldFormat.NPZ:
        return save_npz(field, resolved, options=options)
    if explicit is FieldFormat.MAT:
        return save_mat(field, resolved, options=options)
    if explicit is FieldFormat.ZBF:
        return save_zbf(field, resolved, options=options)
    raise UnsupportedFormatError(f"Unsupported output format: {explicit.value}.")


def _format_from_suffix(path: Path) -> FieldFormat:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return FieldFormat.NPZ
    if suffix == ".mat":
        return FieldFormat.MAT
    if suffix == ".zbf":
        return FieldFormat.ZBF
    raise UnsupportedFormatError(
        f"Unsupported field extension {path.suffix!r}; expected .npz, .mat, or .zbf."
    )


def _coerce_format(value: FieldFormat | str) -> FieldFormat:
    if isinstance(value, FieldFormat):
        return value
    try:
        return FieldFormat(str(value).lower().lstrip("."))
    except (TypeError, ValueError) as exc:
        raise UnsupportedFormatError(
            f"Unsupported field format {value!r}; expected 'npz', 'mat', or 'zbf'."
        ) from exc


__all__ = ["inspect_field", "load_field", "save_field"]
