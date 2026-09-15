"""Explicit conversion of noncanonical array mappings into ``Field2D``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..core import Field2D, Grid2D
from .errors import AmbiguousMappingError, ImportMappingError
from .models import (
    AmplitudeConvention,
    AxisOrder,
    ImportMapping,
    PhaseUnit,
)


def field_from_mapping(
    payload: Mapping[str, Any],
    mapping: ImportMapping,
    *,
    source_path: Path,
) -> Field2D:
    """Apply a fully explicit mapping without guessing units or orientation."""

    if not isinstance(mapping, ImportMapping):
        raise TypeError("mapping must be an ImportMapping.")
    if not mapping.data_keys and not mapping.amplitude_keys:
        raise ImportMappingError(
            "Mapping must select data_keys or amplitude_keys/phase_keys."
        )

    arrays = (
        [_mapped_array(payload, key, mapping) for key in mapping.data_keys]
        if mapping.data_keys
        else _amplitude_phase_arrays(payload, mapping)
    )
    if len(arrays) != len(mapping.components):
        raise ImportMappingError(
            "Mapped array count must equal the output component count."
        )
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ImportMappingError("Mapped component arrays must have identical shapes.")
    if any(not np.isfinite(array.real).all() for array in arrays) or any(
        not np.isfinite(array.imag).all() for array in arrays
    ):
        raise ImportMappingError("Mapped field data contains NaN or infinity.")

    complex_dtype = (
        np.complex128
        if any(array.dtype in (np.dtype("float64"), np.dtype("complex128")) for array in arrays)
        else np.complex64
    )
    data = np.ascontiguousarray(np.stack(arrays).astype(complex_dtype, copy=False))
    if mapping.conjugate:
        data = np.ascontiguousarray(np.conjugate(data))

    dx_m, dy_m = _resolved_spacing(payload, mapping)
    wavelength_m = _resolved_wavelength(payload, mapping)
    medium_index = _resolved_medium_index(payload, mapping)
    x_center_m = _resolved_optional_length(
        payload,
        mapping.x_center_key,
        mapping.x_center_m,
        mapping,
        "x_center",
    )
    y_center_m = _resolved_optional_length(
        payload,
        mapping.y_center_key,
        mapping.y_center_m,
        mapping,
        "y_center",
    )
    z_m = _resolved_optional_length(
        payload,
        mapping.z_key,
        mapping.z_m,
        mapping,
        "z",
    )
    grid = Grid2D(
        nx=data.shape[-1],
        ny=data.shape[-2],
        dx=dx_m,
        dy=dy_m,
        x_center=x_center_m,
        y_center=y_center_m,
    )
    _validate_dimension_metadata(payload, mapping, grid)
    metadata = _mapped_metadata(payload, mapping)
    metadata["import_provenance"] = {
        "source_path": str(Path(source_path).resolve()),
        "axis_order": mapping.axis_order.value,
        "flip_x": mapping.flip_x,
        "flip_y": mapping.flip_y,
    }
    return Field2D(
        data=torch.from_numpy(data),
        grid=grid,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
        components=mapping.components,
        z_m=z_m,
        metadata=metadata,
    )


def _mapped_array(
    payload: Mapping[str, Any],
    key: str,
    mapping: ImportMapping,
) -> np.ndarray:
    value = _numeric_array(payload, key)
    if value.ndim != 2:
        raise ImportMappingError(
            f"Mapped field array {key!r} must be exactly two-dimensional; "
            f"got shape {value.shape}."
        )
    if mapping.axis_order is AxisOrder.XY:
        value = value.T
    if mapping.flip_y:
        value = np.flip(value, axis=0)
    if mapping.flip_x:
        value = np.flip(value, axis=1)
    return np.ascontiguousarray(value)


def _amplitude_phase_arrays(
    payload: Mapping[str, Any],
    mapping: ImportMapping,
) -> list[np.ndarray]:
    output = []
    phase_scale = math.pi / 180.0 if mapping.phase_unit is PhaseUnit.DEG else 1.0
    for amplitude_key, phase_key in zip(
        mapping.amplitude_keys, mapping.phase_keys
    ):
        amplitude = _mapped_array(payload, amplitude_key, mapping)
        phase = _mapped_array(payload, phase_key, mapping)
        if amplitude.shape != phase.shape:
            raise ImportMappingError(
                f"Amplitude {amplitude_key!r} and phase {phase_key!r} "
                "must have identical shapes."
            )
        if np.iscomplexobj(amplitude) or np.iscomplexobj(phase):
            raise ImportMappingError("Amplitude and phase arrays must be real.")
        if not np.isfinite(amplitude).all() or not np.isfinite(phase).all():
            raise ImportMappingError("Amplitude or phase contains NaN or infinity.")
        if np.any(amplitude < 0):
            label = (
                "intensity"
                if mapping.amplitude_convention is AmplitudeConvention.INTENSITY
                else "amplitude"
            )
            raise ImportMappingError(f"Mapped {label} must be non-negative.")
        if mapping.amplitude_convention is AmplitudeConvention.INTENSITY:
            amplitude = np.sqrt(amplitude)
        output.append(
            amplitude
            * np.exp(1j * mapping.phase_sign * phase * phase_scale)
        )
    return output


def _numeric_array(payload: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise ImportMappingError(f"Mapped key {key!r} is missing.")
    value = np.asarray(payload[key])
    if value.dtype.hasobject or value.dtype.fields is not None:
        raise ImportMappingError(f"Mapped key {key!r} is not a plain numeric array.")
    if value.dtype.kind not in "biufc":
        raise ImportMappingError(f"Mapped key {key!r} must be numeric.")
    return value


def _scalar(payload: Mapping[str, Any], key: str, label: str) -> float:
    value = _numeric_array(payload, key)
    if value.size != 1:
        raise ImportMappingError(f"{label} key {key!r} must contain one scalar.")
    scalar = value.reshape(-1)[0]
    if np.iscomplexobj(scalar) and complex(scalar).imag != 0:
        raise ImportMappingError(f"{label} key {key!r} must be real.")
    result = float(np.real(scalar))
    if not math.isfinite(result):
        raise ImportMappingError(f"{label} must be finite.")
    return result


def _resolved_spacing(
    payload: Mapping[str, Any],
    mapping: ImportMapping,
) -> tuple[float, float]:
    if mapping.dx_m is not None or mapping.dy_m is not None:
        if mapping.dx_m is None or mapping.dy_m is None:
            raise ImportMappingError("Both dx_m and dy_m overrides are required.")
        return mapping.dx_m, mapping.dy_m
    if mapping.pixel_size_key is not None:
        scale = _required_spatial_scale(mapping)
        spacing = _scalar(
            payload, mapping.pixel_size_key, "pixel spacing"
        ) * scale
        return _positive(spacing, "pixel spacing"), _positive(
            spacing, "pixel spacing"
        )
    if mapping.dx_key is None or mapping.dy_key is None:
        raise AmbiguousMappingError(
            "Explicit dx/dy values or spacing keys are required."
        )
    scale = _required_spatial_scale(mapping)
    return (
        _positive(_scalar(payload, mapping.dx_key, "dx") * scale, "dx"),
        _positive(_scalar(payload, mapping.dy_key, "dy") * scale, "dy"),
    )


def _resolved_wavelength(
    payload: Mapping[str, Any],
    mapping: ImportMapping,
) -> float:
    if mapping.wavelength_m is not None:
        return mapping.wavelength_m
    if mapping.wavelength_key is None:
        raise AmbiguousMappingError(
            "An explicit wavelength_m or wavelength key/unit is required."
        )
    if mapping.wavelength_unit is None:
        raise AmbiguousMappingError(
            "wavelength_unit is required for an unsuffixed legacy key."
        )
    return _positive(
        _scalar(payload, mapping.wavelength_key, "wavelength")
        * mapping.wavelength_unit.scale_to_m,
        "wavelength",
    )


def _resolved_medium_index(
    payload: Mapping[str, Any],
    mapping: ImportMapping,
) -> complex:
    if mapping.medium_index is not None:
        return mapping.medium_index
    if mapping.medium_index_real_key is None:
        raise AmbiguousMappingError(
            "An explicit medium_index or medium-index key mapping is required."
        )
    real = _scalar(payload, mapping.medium_index_real_key, "medium index real")
    imag = (
        0.0
        if mapping.medium_index_imag_key is None
        else _scalar(payload, mapping.medium_index_imag_key, "medium index imag")
    )
    value = complex(real, imag)
    if value.real <= 0:
        raise ImportMappingError("medium_index.real must be positive.")
    return value


def _resolved_optional_length(
    payload: Mapping[str, Any],
    key: str | None,
    fallback_m: float,
    mapping: ImportMapping,
    label: str,
) -> float:
    if key is None:
        return fallback_m
    scale = _required_spatial_scale(mapping)
    return _scalar(payload, key, label) * scale


def _required_spatial_scale(mapping: ImportMapping) -> float:
    if mapping.spatial_unit is None:
        raise AmbiguousMappingError(
            "spatial_unit is required for legacy spacing/position keys."
        )
    return mapping.spatial_unit.scale_to_m


def _validate_dimension_metadata(
    payload: Mapping[str, Any],
    mapping: ImportMapping,
    grid: Grid2D,
) -> None:
    if mapping.nx_key is not None:
        nx = _scalar(payload, mapping.nx_key, "nx")
        if not nx.is_integer() or int(nx) != grid.nx:
            raise ImportMappingError(
                f"Mapped nx metadata {nx!r} does not match field nx={grid.nx}."
            )
    if mapping.ny_key is not None:
        ny = _scalar(payload, mapping.ny_key, "ny")
        if not ny.is_integer() or int(ny) != grid.ny:
            raise ImportMappingError(
                f"Mapped ny metadata {ny!r} does not match field ny={grid.ny}."
            )


def _mapped_metadata(
    payload: Mapping[str, Any],
    mapping: ImportMapping,
) -> dict[str, Any]:
    if mapping.metadata_key is None:
        return {}
    if mapping.metadata_key not in payload:
        raise ImportMappingError(
            f"Mapped metadata key {mapping.metadata_key!r} is missing."
        )
    raw = np.asarray(payload[mapping.metadata_key])
    if raw.size != 1:
        raise ImportMappingError("Mapped metadata must contain one JSON string.")
    value = raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        metadata = json.loads(str(value), parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ImportMappingError("Mapped metadata is not valid strict JSON.") from exc
    if not isinstance(metadata, dict):
        raise ImportMappingError("Mapped metadata JSON root must be an object.")
    return metadata


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")


def _positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ImportMappingError(f"{label} must be finite and positive.")
    return value


__all__ = ["field_from_mapping"]
