"""Schema-light property form for source and layer dictionaries."""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)


ENUMS = {
    "kind": ["plane_wave", "tilted_plane_wave", "gaussian", "elliptical_gaussian", "imported_field"],
    "method": ["asm", "blas", "fresnel_tf", "fresnel_scaled", "rs_fft", "rs_direct"],
    "phase_model": ["exact_equal_path", "paraxial"],
    "shape": ["circle", "rectangle", "ellipse"],
    "sampling_mode": ["strict", "resample_complex"],
    "padding": ["none", "auto"],
}

HIDDEN_KEYS = {"id", "type", "mapping"}


class PropertyEditor(QWidget):
    """Edit primitive dictionary values and emit one validated snapshot."""

    valuesChanged = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._editors: dict[str, QWidget] = {}
        self._values: dict[str, Any] = {}

    def set_values(self, values: dict[str, Any]) -> None:
        self._values = dict(values)
        while self._form.rowCount():
            self._form.removeRow(0)
        self._editors.clear()
        for key, value in values.items():
            if key in HIDDEN_KEYS or isinstance(value, (dict, list, tuple)):
                continue
            editor = self._make_editor(key, value)
            self._editors[key] = editor
            self._form.addRow(key.replace("_", " "), editor)

    def values(self) -> dict[str, Any]:
        result = dict(self._values)
        for key, editor in self._editors.items():
            if isinstance(editor, QCheckBox):
                result[key] = editor.isChecked()
            elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
                result[key] = editor.value()
            elif isinstance(editor, QComboBox):
                result[key] = editor.currentData()
            else:
                text = editor.text().strip()
                result[key] = None if text.lower() == "none" else text
        return result

    def _make_editor(self, key: str, value: Any) -> QWidget:
        if key in ENUMS:
            editor = QComboBox(self)
            for item in ENUMS[key]:
                editor.addItem(item.replace("_", " ").title(), item)
            editor.setCurrentIndex(max(0, editor.findData(value)))
            editor.currentIndexChanged.connect(self._emit)
            return editor
        if isinstance(value, bool):
            editor = QCheckBox(self)
            editor.setChecked(value)
            editor.toggled.connect(self._emit)
            return editor
        if isinstance(value, int):
            editor = QSpinBox(self)
            editor.setRange(1 if key in {"nx", "ny"} else -1_000_000_000, 1_000_000_000)
            editor.setValue(value)
            editor.valueChanged.connect(self._emit)
            return editor
        if isinstance(value, float):
            editor = QDoubleSpinBox(self)
            editor.setDecimals(12)
            editor.setRange(-1e12, 1e12)
            editor.setSingleStep(max(abs(value) / 10, 1e-9))
            editor.setValue(value)
            editor.valueChanged.connect(self._emit)
            return editor
        editor = QLineEdit("" if value is None else str(value), self)
        editor.editingFinished.connect(self._emit)
        return editor

    def _emit(self, *_args) -> None:
        self.valuesChanged.emit(self.values())


__all__ = ["PropertyEditor"]
