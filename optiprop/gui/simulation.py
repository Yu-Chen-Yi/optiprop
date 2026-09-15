"""Background execution, cancellation, caching, and XZ scan support."""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


@dataclass(frozen=True)
class SimulationOutput:
    result: Any
    xz_intensity: np.ndarray | None = None
    x_axis_m: np.ndarray | None = None
    z_axis_m: np.ndarray | None = None
    cache_hit: bool = False


class SimulationCache:
    """Small thread-safe LRU cache keyed by complete simulation settings."""

    def __init__(self, maximum: int = 4) -> None:
        self.maximum = max(1, int(maximum))
        self._items: OrderedDict[str, SimulationOutput] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> SimulationOutput | None:
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
            return item

    def put(self, key: str, value: SimulationOutput) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.maximum:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class WorkerSignals(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str, str)
    cancelled = Signal(str)
    finished = Signal()


class SimulationWorker(QRunnable):
    def __init__(self, config: dict[str, Any], cache: SimulationCache | None = None) -> None:
        super().__init__()
        from optiprop.propagation import CancellationToken

        self.config = deepcopy(config)
        self.cache = cache
        self.signals = WorkerSignals()
        self.token = CancellationToken()
        self.setAutoDelete(True)

    def cancel(self, reason: str = "Cancelled by user") -> None:
        self.token.cancel(reason)

    def run(self) -> None:
        try:
            key = simulation_key(self.config)
            cached = None if self.cache is None else self.cache.get(key)
            if cached is not None:
                self.signals.progress.emit(100, "Loaded cached result")
                self.signals.result.emit(SimulationOutput(
                    cached.result, cached.xz_intensity, cached.x_axis_m,
                    cached.z_axis_m, True,
                ))
                return
            self.signals.progress.emit(3, "Building incident source")
            input_field = build_source(self.config.get("source", {})).create()
            self.token.throw_if_cancelled()
            self.signals.progress.emit(12, "Building optical system")
            system = build_system(self.config.get("layers", []))
            self.signals.progress.emit(20, "Propagating layers")
            result = system.execute(input_field, cancel_token=self.token)
            self.signals.progress.emit(78, "Preparing result maps")
            # XZ is an observation after the configured stack, matching the
            # result tabs and the physical meaning of "after metalens".
            xz_data, x_axis, z_axis = self._xz_scan(result.output_field)
            output = SimulationOutput(result, xz_data, x_axis, z_axis)
            if self.cache is not None:
                self.cache.put(key, output)
            self.signals.progress.emit(100, "Complete")
            self.signals.result.emit(output)
        except Exception as exc:
            from optiprop.propagation import PropagationCancelled

            if isinstance(exc, PropagationCancelled):
                self.signals.cancelled.emit(str(exc))
            else:
                self.signals.error.emit(str(exc), traceback.format_exc())
        finally:
            self.signals.finished.emit()

    def _xz_scan(self, field):
        settings = self.config.get("xz", {})
        if not settings.get("enabled", False):
            return None, None, None
        from optiprop.propagation import PropagationSpec
        from optiprop.system import default_backend_registry

        count = max(2, min(int(settings.get("count", 64)), 1000))
        z_values = np.linspace(
            float(settings.get("z_start_m", 0.0)),
            float(settings.get("z_stop_m", 0.02)), count,
        )
        method = settings.get("method", "asm")
        backend = default_backend_registry().resolve(method)
        rows: list[np.ndarray] = []
        for index, z_value in enumerate(z_values):
            self.token.throw_if_cancelled()
            if math.isclose(float(z_value), 0.0, abs_tol=1e-18):
                propagated = field
            else:
                propagated = backend.propagate(
                    field,
                    PropagationSpec(method=method, distance_m=float(z_value)),
                    cancel_token=self.token,
                ).field
            intensity = propagated.intensity().detach().cpu().numpy()
            rows.append(intensity[intensity.shape[0] // 2])
            self.signals.progress.emit(
                78 + int(20 * (index + 1) / count), f"XZ scan {index + 1}/{count}"
            )
        return np.stack(rows), field.grid.x_coordinates().cpu().numpy(), z_values


class SimulationService(QObject):
    started = Signal()
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str, str)
    cancelled = Signal(str)
    finished = Signal()

    def __init__(self, parent=None, *, cache_size: int = 4) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.cache = SimulationCache(cache_size)
        self.worker: SimulationWorker | None = None

    @property
    def is_running(self) -> bool:
        return self.worker is not None

    def start(self, config: dict[str, Any]) -> bool:
        if self.worker is not None:
            return False
        worker = SimulationWorker(config, self.cache)
        self.worker = worker
        worker.signals.progress.connect(self.progress)
        worker.signals.result.connect(self.result)
        worker.signals.error.connect(self.error)
        worker.signals.cancelled.connect(self.cancelled)
        worker.signals.finished.connect(self._finished)
        self.started.emit()
        self.pool.start(worker)
        return True

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def _finished(self) -> None:
        self.worker = None
        self.finished.emit()


def simulation_key(config: dict[str, Any]) -> str:
    normalized = deepcopy(config)
    sections = [normalized.get("source", {})] + list(normalized.get("layers", []))
    for section in sections:
        path = section.get("path") if isinstance(section, dict) else None
        if path:
            candidate = Path(path)
            if candidate.is_file():
                stat = candidate.stat()
                section["_asset_state"] = [stat.st_size, stat.st_mtime_ns]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _complex(value: Any) -> complex:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(value[0], value[1])
    if isinstance(value, dict) and set(value) >= {"real", "imag"}:
        return complex(value["real"], value["imag"])
    return complex(value)


def _mapping(config: dict[str, Any]):
    if not config:
        return None
    from optiprop.io.models import ImportMapping

    values = dict(config)
    for key in ("data_keys", "amplitude_keys", "phase_keys", "components"):
        if key in values:
            values[key] = tuple(values[key])
    return ImportMapping(**values)


def build_source(config: dict[str, Any]):
    from optiprop.core import Grid2D
    from optiprop.gui.models import DEFAULT_SOURCE
    from optiprop.io import load_field
    from optiprop.system import IncidentSource

    values = {**deepcopy(DEFAULT_SOURCE), **deepcopy(config)}
    kind = values["kind"]
    if kind == "imported_field":
        path = values.get("path", "")
        if not path:
            raise ValueError("Imported source requires a field path.")
        field = load_field(path, _mapping(values.get("mapping", {})))
        return IncidentSource.from_field(field, provenance={"path": str(Path(path).resolve())})
    grid = Grid2D(int(values["nx"]), int(values["ny"]), float(values["dx_m"]), float(values["dy_m"]))
    common = dict(
        grid=grid, wavelength_m=float(values["wavelength_m"]),
        medium_index=_complex(values.get("medium_index", 1.0)),
        amplitudes=(_complex(values.get("amplitude", 1.0)),),
        phase_offset_rad=float(values.get("phase_offset_rad", 0.0)),
    )
    if kind in {"gaussian", "elliptical_gaussian"}:
        return IncidentSource.gaussian(
            waist_x_m=float(values["waist_x_m"]),
            waist_y_m=float(values["waist_y_m"]), **common,
        )
    if kind == "tilted_plane_wave":
        return IncidentSource.tilted_plane_wave(
            angle_x_rad=float(values.get("angle_x_rad", 0.0)),
            angle_y_rad=float(values.get("angle_y_rad", 0.0)), **common,
        )
    return IncidentSource.plane_wave(**common)


def build_system(configs: list[dict[str, Any]]):
    from optiprop.io import load_field
    from optiprop.propagation import PaddingSpec, PropagationSpec
    from optiprop.system import (
        ApertureLayer, ApertureSpec, ComplexMaskLayer, IdealLensLayer,
        InterfaceLayer, OpticalSystem, PropagationLayer,
    )

    layers = []
    for item in configs:
        values = deepcopy(item)
        common = {
            "id": UUID(values["id"]), "name": values.get("name", values["type"]),
            "enabled": bool(values.get("enabled", True)),
        }
        kind = values["type"]
        if kind == "propagation":
            padding = PaddingSpec.none() if values.get("padding") == "none" else PaddingSpec.auto()
            spec = PropagationSpec(
                method=values.get("method", "asm"),
                distance_m=float(values["distance_m"]),
                medium_index=_complex(values.get("medium_index", 1.0)), padding=padding,
            )
            layers.append(PropagationLayer(spec=spec, **common))
        elif kind == "lens":
            layers.append(IdealLensLayer(
                focal_length_m=float(values["focal_length_m"]),
                phase_model=values.get("phase_model", "exact_equal_path"),
                transmission_amplitude=float(values.get("transmission_amplitude", 1.0)), **common,
                design_wavelength_m=(
                    None
                    if values.get("design_wavelength_m") in (None, "")
                    else float(values["design_wavelength_m"])
                ),
                medium_index=(
                    None
                    if values.get("medium_index") in (None, "")
                    else _complex(values["medium_index"])
                ),
            ))
        elif kind == "aperture":
            aperture = ApertureSpec(
                shape=values.get("shape", "circle"), size_x_m=float(values["size_x_m"]),
                size_y_m=float(values.get("size_y_m", values["size_x_m"])),
            )
            layers.append(ApertureLayer(aperture=aperture, **common))
        elif kind == "interface":
            layers.append(InterfaceLayer(
                n1=_complex(values["n1"]), n2=_complex(values["n2"]),
                ignore_reflection=bool(values.get("ignore_reflection", False)), **common,
            ))
        elif kind == "mask":
            if not values.get("path"):
                raise ValueError("Complex mask requires a field path.")
            field = load_field(values["path"], _mapping(values.get("mapping", {})))
            layers.append(ComplexMaskLayer(
                transmission=field.data, grid=field.grid,
                sampling_mode=values.get("sampling_mode", "strict"),
                asset_reference=str(Path(values["path"]).resolve()), **common,
            ))
        else:
            raise ValueError(f"Unsupported layer type: {kind}")
    return OpticalSystem(tuple(layers), name="GUI optical system")


__all__ = [
    "SimulationCache", "SimulationOutput", "SimulationService", "SimulationWorker",
    "build_source", "build_system", "simulation_key",
]
