"""Field import inspection and explicit mapping dialog."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class FieldImportDialog(QDialog):
    """Inspect NPZ/MAT/ZBF and capture noncanonical amplitude/phase mapping."""

    def __init__(self, path: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import optical field")
        self.resize(620, 430)
        self.path_edit = QLineEdit(path, self)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        inspect = QPushButton("Inspect", self)
        inspect.clicked.connect(self.inspect)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        path_row.addWidget(inspect)

        self.status = QLabel("Choose a .npz, .mat, or .zbf field.", self)
        self.status.setWordWrap(True)
        self.data_key = QComboBox(self)
        self.amplitude_key = QComboBox(self)
        self.phase_key = QComboBox(self)
        self.dx = QLineEdit("", self)
        self.dy = QLineEdit("", self)
        self.wavelength = QLineEdit("", self)
        self.axis = QComboBox(self)
        self.axis.addItems(["yx", "xy"])
        self.phase_unit = QComboBox(self)
        self.phase_unit.addItems(["rad", "deg"])
        self.flip_x = QCheckBox(self)
        self.flip_y = QCheckBox(self)
        form = QFormLayout()
        form.addRow("Complex data key", self.data_key)
        form.addRow("Amplitude key", self.amplitude_key)
        form.addRow("Phase key", self.phase_key)
        form.addRow("dx [m]", self.dx)
        form.addRow("dy [m]", self.dy)
        form.addRow("Wavelength [m]", self.wavelength)
        form.addRow("Array axis order", self.axis)
        form.addRow("Phase unit", self.phase_unit)
        form.addRow("Flip X", self.flip_x)
        form.addRow("Flip Y", self.flip_y)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(path_row)
        layout.addWidget(self.status)
        layout.addLayout(form)
        layout.addStretch()
        layout.addWidget(buttons)
        if path:
            self.inspect()

    def inspect(self) -> None:
        path = self.path_edit.text().strip()
        try:
            from optiprop.io import inspect_field

            info = inspect_field(path)
        except Exception as exc:
            self.status.setText(f"Inspection failed: {exc}")
            return
        keys = list(info.keys)
        for combo in (self.data_key, self.amplitude_key, self.phase_key):
            combo.clear()
            combo.addItem("(none)", "")
            for key in keys:
                combo.addItem(key, key)
        suggestion = info.suggested_mapping
        if suggestion is not None:
            self._select(self.data_key, suggestion.data_keys[0] if suggestion.data_keys else "")
            self._select(self.amplitude_key, suggestion.amplitude_keys[0] if suggestion.amplitude_keys else "")
            self._select(self.phase_key, suggestion.phase_keys[0] if suggestion.phase_keys else "")
        self.status.setText(
            f"{info.format.value.upper()} · {len(info.arrays)} arrays · "
            f"{'canonical' if info.is_canonical else info.mapping_confidence.value}"
        )

    @property
    def import_config(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "axis_order": self.axis.currentText(),
            "phase_unit": self.phase_unit.currentText(),
            "flip_x": self.flip_x.isChecked(),
            "flip_y": self.flip_y.isChecked(),
        }
        data = self.data_key.currentData()
        amplitude = self.amplitude_key.currentData()
        phase = self.phase_key.currentData()
        if data:
            mapping["data_keys"] = [data]
        elif amplitude and phase:
            mapping["amplitude_keys"] = [amplitude]
            mapping["phase_keys"] = [phase]
        for key, editor in (("dx_m", self.dx), ("dy_m", self.dy), ("wavelength_m", self.wavelength)):
            if editor.text().strip():
                mapping[key] = float(editor.text())
        return {"path": str(Path(self.path_edit.text().strip()).expanduser()), "mapping": mapping}

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import optical field", "", "Optical fields (*.npz *.mat *.zbf)")
        if path:
            self.path_edit.setText(path)
            self.inspect()

    def _accept_if_valid(self) -> None:
        path = Path(self.path_edit.text().strip()).expanduser()
        if not path.is_file() or path.suffix.lower() not in {".npz", ".mat", ".zbf"}:
            QMessageBox.warning(self, "Invalid field", "Choose an existing NPZ, MAT, or ZBF file.")
            return
        try:
            self.import_config
        except ValueError:
            QMessageBox.warning(self, "Invalid mapping", "dx, dy, and wavelength must be numbers in metres.")
            return
        self.accept()

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)


__all__ = ["FieldImportDialog"]
