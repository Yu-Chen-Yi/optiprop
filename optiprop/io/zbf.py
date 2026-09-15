"""Strict Zemax beam-file (ZBF) v0/v1 input and v1 output.

ZBF stores little-endian samples with X varying fastest.  Public OptiProp
objects always use SI lengths and ``[component, y, x]`` array ordering.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import struct
import tempfile

import numpy as np
import torch

from ..core import EX_EY_COMPONENTS, SCALAR_COMPONENTS, Field2D, Grid2D, Severity, ValidationIssue
from .errors import ImportMappingError, SchemaValidationError, UnsafeDatasetError
from .models import (
    ArrayInspection,
    DatasetInspection,
    ExportOptions,
    ExportReport,
    FieldFormat,
    ImportMapping,
    MappingConfidence,
)


SCHEMA_NAME = "zemax.zbf"
SCHEMA_VERSION = 1
_HEADER_SIZE = 9 * 4
_V0_DOUBLE_COUNT = 11
_V1_DOUBLE_COUNT = 20
_UNIT_TO_METRE = {0: 1e-3, 1: 1e-2, 2: 0.0254, 3: 1.0}
_UNIT_NAMES = {0: "mm", 1: "cm", 2: "in", 3: "m"}
_MAX_FILE_BYTES = 2 * 1024**3
_MAX_SAMPLES = 100_000_000


@dataclass(frozen=True)
class ZbfPilotRays:
    """Pilot-ray values, expressed in metres."""

    position_x_m: float = 0.0
    rayleigh_x_m: float = 0.0
    waist_x_m: float = 0.0
    position_y_m: float = 0.0
    rayleigh_y_m: float = 0.0
    waist_y_m: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ZbfBeam:
    """Validated in-memory ZBF beam with all physical lengths in SI."""

    ex: np.ndarray
    dx_m: float
    dy_m: float
    wavelength_m: float
    ey: np.ndarray | None = None
    medium_index: float = 1.0
    pilot: ZbfPilotRays = field(default_factory=ZbfPilotRays)
    version: int = 1
    source_unit: int = 3

    def __post_init__(self) -> None:
        ex = np.asarray(self.ex)
        if ex.ndim != 2 or ex.size == 0:
            raise ValueError("ex must be a non-empty two-dimensional array.")
        if ex.size > _MAX_SAMPLES:
            raise UnsafeDatasetError(
                f"ZBF beam has {ex.size} samples; limit is {_MAX_SAMPLES}."
            )
        if not np.issubdtype(ex.dtype, np.complexfloating):
            ex = ex.astype(np.complex128)
        ey = None if self.ey is None else np.asarray(self.ey)
        if ey is not None:
            if ey.shape != ex.shape:
                raise ValueError("ey must have the same shape as ex.")
            if not np.issubdtype(ey.dtype, np.complexfloating):
                ey = ey.astype(np.complex128)
        encoded_size = _HEADER_SIZE + _V1_DOUBLE_COUNT * 8 + ex.size * 16 * (
            2 if ey is not None else 1
        )
        if encoded_size > _MAX_FILE_BYTES:
            raise UnsafeDatasetError(
                f"Encoded ZBF would be {encoded_size} bytes; limit is "
                f"{_MAX_FILE_BYTES} bytes."
            )
        for name in ("dx_m", "dy_m", "wavelength_m", "medium_index"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
            object.__setattr__(self, name, value)
        if self.version not in (0, 1):
            raise ValueError("version must be 0 or 1.")
        if self.source_unit not in _UNIT_TO_METRE:
            raise ValueError("source_unit must be a supported ZBF unit code.")
        object.__setattr__(self, "ex", ex)
        object.__setattr__(self, "ey", ey)

    @property
    def ny(self) -> int:
        return int(self.ex.shape[0])

    @property
    def nx(self) -> int:
        return int(self.ex.shape[1])

    @property
    def is_polarized(self) -> bool:
        return self.ey is not None

    def to_field(self, *, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> Field2D:
        arrays = [self.ex] if self.ey is None else [self.ex, self.ey]
        data = torch.from_numpy(np.stack(arrays).copy())
        if dtype is not None and dtype not in (torch.complex64, torch.complex128):
            raise TypeError("dtype must be torch.complex64 or torch.complex128.")
        data = data.to(device=device, dtype=dtype)
        return Field2D(
            data=data,
            grid=Grid2D(nx=self.nx, ny=self.ny, dx=self.dx_m, dy=self.dy_m),
            wavelength_m=self.wavelength_m,
            medium_index=self.medium_index,
            components=SCALAR_COMPONENTS if self.ey is None else EX_EY_COMPONENTS,
            metadata={
                "source_format": "zbf",
                "zbf_version": self.version,
                "zbf_source_unit": _UNIT_NAMES[self.source_unit],
                "zbf_pilot_rays_m": {
                    name: getattr(self.pilot, name)
                    for name in self.pilot.__dataclass_fields__
                },
            },
        )


def read_zbf(path: str | os.PathLike[str], *, strict: bool = True) -> ZbfBeam:
    """Read ZBF v0/v1 after validating header, size, and field payload."""

    source = _existing_zbf(path)
    size = source.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise UnsafeDatasetError(
            f"ZBF file is {size} bytes; limit is {_MAX_FILE_BYTES} bytes."
        )
    with source.open("rb") as handle:
        raw_header = handle.read(_HEADER_SIZE)
        if len(raw_header) != _HEADER_SIZE:
            raise SchemaValidationError("File is too short for a ZBF header.")
        version, nx, ny, ispol, unit, *reserved = struct.unpack("<9i", raw_header)
        if version not in (0, 1):
            raise SchemaValidationError(f"Unsupported ZBF version {version}; expected 0 or 1.")
        if nx <= 0 or ny <= 0:
            raise SchemaValidationError("ZBF nx and ny must be positive.")
        samples = nx * ny
        if samples > _MAX_SAMPLES:
            raise UnsafeDatasetError(
                f"ZBF declares {samples} samples; limit is {_MAX_SAMPLES}."
            )
        if ispol not in (0, 1):
            raise SchemaValidationError("ZBF polarization flag must be 0 or 1.")
        if unit not in _UNIT_TO_METRE:
            raise SchemaValidationError(f"Unsupported ZBF unit code {unit}.")
        if strict and any(reserved):
            raise SchemaValidationError("Reserved ZBF header integers must be zero.")

        count = _V1_DOUBLE_COUNT if version == 1 else _V0_DOUBLE_COUNT
        meta_bytes = handle.read(count * 8)
        if len(meta_bytes) != count * 8:
            raise SchemaValidationError("File is truncated in the ZBF numeric header.")
        values = struct.unpack(f"<{count}d", meta_bytes)
        scale = _UNIT_TO_METRE[unit]
        if version == 1:
            dx, dy, px, rx, wx, py, ry, wy, wavelength, index = values[:10]
        else:
            dx, dy, position, rayleigh, wavelength, waist = values[:6]
            px = py = position
            rx = ry = rayleigh
            wx = wy = waist
            index = 1.0
        named = {"dx": dx, "dy": dy, "wavelength": wavelength, "index": index}
        if any(not math.isfinite(v) or v <= 0 for v in named.values()):
            raise SchemaValidationError(
                "ZBF dx, dy, wavelength, and refractive index must be positive and finite."
            )
        pilots = (px, rx, wx, py, ry, wy)
        if any(not math.isfinite(v) for v in pilots):
            raise SchemaValidationError("ZBF pilot-ray values must be finite.")

        field_bytes = samples * 16
        expected_size = _HEADER_SIZE + count * 8 + field_bytes * (1 + ispol)
        if size < expected_size:
            raise SchemaValidationError(
                f"ZBF field payload is truncated: expected {expected_size} bytes, got {size}."
            )
        if strict and size != expected_size:
            raise SchemaValidationError(
                f"ZBF has {size - expected_size} unexpected trailing byte(s)."
            )
        ex = _read_component(handle, nx, ny)
        ey = _read_component(handle, nx, ny) if ispol else None

    if strict and (not np.isfinite(ex).all() or (ey is not None and not np.isfinite(ey).all())):
        raise SchemaValidationError("ZBF field contains NaN or infinity.")
    return ZbfBeam(
        ex=ex, ey=ey, dx_m=dx * scale, dy_m=dy * scale,
        wavelength_m=wavelength * scale, medium_index=index,
        pilot=ZbfPilotRays(*(value * scale for value in pilots)),
        version=version, source_unit=unit,
    )


def inspect_zbf(path: str | os.PathLike[str], *, strict: bool = True) -> DatasetInspection:
    """Inspect a self-describing ZBF and return canonical SI metadata."""

    source = _existing_zbf(path)
    beam = read_zbf(source, strict=strict)
    arrays = [_inspect_component("Ex", beam.ex)]
    if beam.ey is not None:
        arrays.append(_inspect_component("Ey", beam.ey))
    return DatasetInspection(
        path=source, format=FieldFormat.ZBF,
        keys=tuple(item.key for item in arrays), arrays=tuple(arrays),
        scalar_metadata={
            "nx": beam.nx, "ny": beam.ny, "dx_m": beam.dx_m,
            "dy_m": beam.dy_m, "wavelength_m": beam.wavelength_m,
            "medium_index": beam.medium_index,
            "components": ("Ex", "Ey") if beam.is_polarized else ("scalar",),
        },
        schema_name=SCHEMA_NAME, schema_version=beam.version,
        is_canonical=True, mapping_confidence=MappingConfidence.EXACT,
    )


def load_zbf(
    path: str | os.PathLike[str], mapping: ImportMapping | None = None, *,
    strict: bool = True, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Field2D:
    """Load a ZBF as a canonical :class:`Field2D`."""

    if mapping is not None:
        raise ImportMappingError("ZBF is self-describing and does not accept ImportMapping.")
    return read_zbf(path, strict=strict).to_field(device=device, dtype=dtype)


def write_zbf(
    beam: ZbfBeam, path: str | os.PathLike[str], *, atomic: bool = True,
) -> None:
    """Write an in-memory beam as ZBF v1 using metre units."""

    if not isinstance(beam, ZbfBeam):
        raise TypeError("beam must be a ZbfBeam.")
    destination = _destination(path)
    if not np.isfinite(beam.ex).all() or (beam.ey is not None and not np.isfinite(beam.ey).all()):
        raise SchemaValidationError("Cannot export a ZBF field containing NaN/Inf.")
    payload = _encode_v1(beam)
    _write_bytes(payload, destination, atomic=atomic)


def save_zbf(
    field: Field2D, path: str | os.PathLike[str], *,
    options: ExportOptions | None = None,
) -> ExportReport:
    """Export a scalar or Ex/Ey field as loss-aware ZBF v1."""

    if not isinstance(field, Field2D):
        raise TypeError("field must be a Field2D.")
    options = ExportOptions() if options is None else options
    if not isinstance(options, ExportOptions):
        raise TypeError("options must be an ExportOptions or None.")
    if abs(field.medium_index.imag) > 0:
        raise SchemaValidationError(
            "ZBF stores only a real refractive index; absorbing media cannot be exported losslessly."
        )
    data = field.data.detach().to(device="cpu", dtype=torch.complex128).numpy()
    beam = ZbfBeam(
        ex=np.array(data[0], copy=True),
        ey=None if field.is_scalar else np.array(data[1], copy=True),
        dx_m=field.grid.dx, dy_m=field.grid.dy,
        wavelength_m=field.wavelength_m, medium_index=field.medium_index.real,
    )
    destination = _destination(path)
    write_zbf(beam, destination, atomic=options.atomic)
    warnings: list[ValidationIssue] = []
    if field.grid.x_center != 0 or field.grid.y_center != 0 or field.z_m != 0:
        warnings.append(ValidationIssue(
            severity=Severity.WARNING, code="io.zbf.unstored_coordinates",
            message="ZBF does not store grid centre or absolute z; these coordinates were omitted.",
        ))
    return ExportReport(
        path=destination, format=FieldFormat.ZBF, schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION, bytes_written=destination.stat().st_size,
        sha256=_sha256(destination), atomic=options.atomic, compressed=False,
        warnings=tuple(warnings),
    )


def _read_component(handle, nx: int, ny: int) -> np.ndarray:
    raw = handle.read(nx * ny * 16)
    if len(raw) != nx * ny * 16:
        raise SchemaValidationError("File ended before the complete ZBF field was read.")
    pairs = np.frombuffer(raw, dtype="<f8").reshape(-1, 2)
    return (pairs[:, 0] + 1j * pairs[:, 1]).reshape(ny, nx)


def _inspect_component(key: str, value: np.ndarray) -> ArrayInspection:
    magnitude = np.abs(value)
    return ArrayInspection(
        key=key, shape=tuple(value.shape), dtype=str(value.dtype),
        is_complex=True, is_object=False, is_finite=bool(np.isfinite(value).all()),
        magnitude_min=float(magnitude.min()), magnitude_max=float(magnitude.max()),
    )


def _encode_v1(beam: ZbfBeam) -> bytes:
    header = struct.pack(
        "<9i", 1, beam.nx, beam.ny, int(beam.is_polarized), 3, 0, 0, 0, 0
    )
    pilot = beam.pilot
    values = [
        beam.dx_m, beam.dy_m, pilot.position_x_m, pilot.rayleigh_x_m,
        pilot.waist_x_m, pilot.position_y_m, pilot.rayleigh_y_m,
        pilot.waist_y_m, beam.wavelength_m, beam.medium_index,
    ] + [0.0] * 10
    chunks = [header, struct.pack("<20d", *values), _component_bytes(beam.ex)]
    if beam.ey is not None:
        chunks.append(_component_bytes(beam.ey))
    return b"".join(chunks)


def _component_bytes(value: np.ndarray) -> bytes:
    flat = np.asarray(value, dtype=np.complex128).ravel(order="C")
    interleaved = np.empty(flat.size * 2, dtype="<f8")
    interleaved[0::2] = flat.real
    interleaved[1::2] = flat.imag
    return interleaved.tobytes()


def _existing_zbf(path: str | os.PathLike[str]) -> Path:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".zbf":
        raise ValueError("ZBF path must end in '.zbf'.")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _destination(path: str | os.PathLike[str]) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".zbf":
        raise ValueError("ZBF export path must end in '.zbf'.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Export directory does not exist: {destination.parent}")
    return destination


def _write_bytes(payload: bytes, destination: Path, *, atomic: bool) -> None:
    temporary: Path | None = None
    try:
        if atomic:
            with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        else:
            with destination.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ZbfBeam", "ZbfPilotRays", "inspect_zbf", "load_zbf", "read_zbf",
    "save_zbf", "write_zbf",
]
