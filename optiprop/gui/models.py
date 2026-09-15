"""Editable, project-independent Qt models used by the desktop UI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable
from uuid import uuid4

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, Signal


LAYER_DEFAULTS: dict[str, dict[str, Any]] = {
    "propagation": {
        "name": "Propagation",
        "method": "asm",
        "distance_m": 0.01,
        "medium_index": 1.0,
        "padding": "none",
    },
    "lens": {
        "name": "Ideal lens",
        "focal_length_m": 0.02,
        "phase_model": "exact_equal_path",
        "transmission_amplitude": 1.0,
    },
    "aperture": {
        "name": "Aperture",
        "shape": "circle",
        "size_x_m": 0.001,
        "size_y_m": 0.001,
    },
    "interface": {
        "name": "Interface",
        "n1": 1.0,
        "n2": 1.5,
        "ignore_reflection": False,
    },
    "mask": {
        "name": "Complex mask",
        "path": "",
        "sampling_mode": "strict",
        "mapping": {},
    },
}

DEFAULT_SOURCE: dict[str, Any] = {
    "kind": "gaussian",
    "nx": 256,
    "ny": 256,
    "dx_m": 5e-6,
    "dy_m": 5e-6,
    "wavelength_m": 532e-9,
    "medium_index": 1.0,
    "amplitude": 1.0,
    "phase_offset_rad": 0.0,
    "waist_x_m": 0.00025,
    "waist_y_m": 0.00025,
    "angle_x_rad": 0.0,
    "angle_y_rad": 0.0,
    "path": "",
    "mapping": {},
}


def new_layer(kind: str) -> dict[str, Any]:
    """Create a serializable layer row with a stable identity."""

    if kind not in LAYER_DEFAULTS:
        raise ValueError(f"Unsupported layer kind: {kind}")
    layer = deepcopy(LAYER_DEFAULTS[kind])
    return {"id": str(uuid4()), "type": kind, "enabled": True, **layer}


class SourceModel(QObject):
    changed = Signal(dict)

    def __init__(self, config: dict[str, Any] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._config = deepcopy(DEFAULT_SOURCE if config is None else config)

    @property
    def config(self) -> dict[str, Any]:
        return deepcopy(self._config)

    def replace(self, config: dict[str, Any]) -> None:
        self._config = deepcopy(config)
        self.changed.emit(self.config)

    def update(self, values: dict[str, Any]) -> None:
        self._config.update(deepcopy(values))
        self.changed.emit(self.config)


class LayerListModel(QAbstractListModel):
    """Mutable UI model whose rows are plain serializable dictionaries."""

    TypeRole = Qt.ItemDataRole.UserRole + 1
    IdRole = Qt.ItemDataRole.UserRole + 2
    ConfigRole = Qt.ItemDataRole.UserRole + 3

    def __init__(self, layers: Iterable[dict[str, Any]] = (), parent=None) -> None:
        super().__init__(parent)
        self._layers = [self._normalized(item) for item in layers]

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._layers)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._layers):
            return None
        item = self._layers[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return item.get("name", item["type"])
        if role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if item["enabled"] else Qt.CheckState.Unchecked
        if role == self.TypeRole:
            return item["type"]
        if role == self.IdRole:
            return item["id"]
        if role == self.ConfigRole:
            return deepcopy(item)
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):  # noqa: N802
        if not index.isValid() or not 0 <= index.row() < len(self._layers):
            return False
        if role == Qt.ItemDataRole.CheckStateRole:
            self._layers[index.row()]["enabled"] = value == Qt.CheckState.Checked.value or value == Qt.CheckState.Checked
        elif role == Qt.ItemDataRole.EditRole:
            self._layers[index.row()]["name"] = str(value).strip() or self._layers[index.row()]["type"]
        else:
            return False
        self.dataChanged.emit(index, index, [role])
        return True

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid():
            flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
        return flags

    def roleNames(self):  # noqa: N802
        return {
            self.TypeRole: b"type",
            self.IdRole: b"id",
            self.ConfigRole: b"config",
        }

    @property
    def layers(self) -> list[dict[str, Any]]:
        return deepcopy(self._layers)

    def layer(self, row: int) -> dict[str, Any]:
        return deepcopy(self._layers[row])

    def replace(self, layers: Iterable[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._layers = [self._normalized(item) for item in layers]
        self.endResetModel()

    def append_kind(self, kind: str) -> int:
        return self.append(new_layer(kind))

    def append(self, layer: dict[str, Any]) -> int:
        item = self._normalized(layer)
        row = len(self._layers)
        self.beginInsertRows(QModelIndex(), row, row)
        self._layers.append(item)
        self.endInsertRows()
        return row

    def remove_row(self, row: int) -> bool:
        if not 0 <= row < len(self._layers):
            return False
        self.beginRemoveRows(QModelIndex(), row, row)
        del self._layers[row]
        self.endRemoveRows()
        return True

    def move(self, row: int, delta: int) -> int:
        target = row + delta
        if not (0 <= row < len(self._layers) and 0 <= target < len(self._layers)):
            return row
        destination = target if target < row else target + 1
        self.beginMoveRows(QModelIndex(), row, row, QModelIndex(), destination)
        self._layers.insert(target, self._layers.pop(row))
        self.endMoveRows()
        return target

    def update_layer(self, row: int, values: dict[str, Any]) -> None:
        if not 0 <= row < len(self._layers):
            raise IndexError(row)
        protected = {"id", "type"}
        self._layers[row].update({k: deepcopy(v) for k, v in values.items() if k not in protected})
        index = self.index(row)
        self.dataChanged.emit(index, index)

    @staticmethod
    def _normalized(layer: dict[str, Any]) -> dict[str, Any]:
        item = deepcopy(layer)
        kind = item.get("type")
        if kind not in LAYER_DEFAULTS:
            raise ValueError(f"Unsupported layer kind: {kind}")
        merged = {"id": str(item.get("id") or uuid4()), "type": kind, "enabled": True}
        merged.update(deepcopy(LAYER_DEFAULTS[kind]))
        merged.update(item)
        merged["enabled"] = bool(merged["enabled"])
        return merged


__all__ = ["DEFAULT_SOURCE", "LAYER_DEFAULTS", "LayerListModel", "SourceModel", "new_layer"]
