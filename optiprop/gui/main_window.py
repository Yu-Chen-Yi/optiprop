"""OptiProp near-field multilayer desktop window."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListView,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QStatusBar, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget,
    QDoubleSpinBox,
)

from .import_dialog import FieldImportDialog
from .models import DEFAULT_SOURCE, LayerListModel, SourceModel
from .plots import ResultPlotWidget
from .property_editor import PropertyEditor
from .simulation import SimulationService, build_source, build_system


class MainWindow(QMainWindow):
    """Project editor and asynchronous near-field simulation workbench."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("OptiPropMainWindow")
        self.setWindowTitle("OptiProp Near-field Workbench")
        self.resize(1280, 800)
        self.project_path: Path | None = None
        self._project_id = uuid4()
        self._project_name = "Untitled project"
        self._project_compute = None
        self._project_assets = ()
        self._project_metadata: dict = {}
        self.source_model = SourceModel()
        self.layer_model = LayerListModel()
        self.service = SimulationService(self)
        self._build_ui()
        self._connect()
        self.source_editor.set_values(self.source_model.config)
        self._restore_settings()

    def project_dict(self) -> dict:
        return {
            "schema": "optiprop-gui-project", "version": 1,
            "source": self.source_model.config, "layers": self.layer_model.layers,
            "xz": {
                "enabled": self.xz_enabled.isChecked(), "method": self.xz_method.currentData(),
                "z_start_m": self.xz_start.value(), "z_stop_m": self.xz_stop.value(),
                "count": self.xz_count.value(),
            },
        }

    def persisted_project(self):
        """Return the versioned, toolkit-independent project document."""

        from optiprop.project import ComputeSettings, OptiPropProject

        runtime = self.project_dict()
        source = deepcopy(runtime["source"])
        layers = deepcopy(runtime["layers"])
        if self.project_path is not None:
            _make_paths_portable(source, self.project_path.parent)
            for layer in layers:
                _make_paths_portable(layer, self.project_path.parent)
        return OptiPropProject(
            id=self._project_id,
            name=self._project_name,
            source=source,
            optical_system={"layers": layers},
            compute_settings=self._project_compute or ComputeSettings(),
            ui_state={"xz": runtime["xz"]},
            assets=self._project_assets,
            metadata=self._project_metadata,
        )

    def load_dict(self, project: dict, *, base_dir: Path | None = None) -> None:
        if project.get("format") == "optiprop-project":
            from optiprop.project import OptiPropProject

            persisted = OptiPropProject.from_dict(project)
            normalized = persisted.to_dict()
            self._project_id = persisted.id
            self._project_name = persisted.name
            self._project_compute = persisted.compute_settings
            self._project_assets = persisted.assets
            self._project_metadata = dict(persisted.metadata)
            source = normalized["source"]
            layers = normalized["optical_system"].get("layers", [])
            xz = normalized["ui_state"].get("xz", {})
        else:
            # Accept WP-11 draft documents so early GUI projects remain usable.
            source = project.get("source", DEFAULT_SOURCE)
            layers = project.get("layers", [])
            xz = project.get("xz", {})
        if base_dir is not None:
            source = deepcopy(source)
            layers = deepcopy(layers)
            _resolve_config_path(source, base_dir)
            for layer in layers:
                _resolve_config_path(layer, base_dir)
        self.source_model.replace(source)
        self.layer_model.replace(layers)
        self.xz_enabled.setChecked(bool(xz.get("enabled", False)))
        index = self.xz_method.findData(xz.get("method", "asm"))
        self.xz_method.setCurrentIndex(max(0, index))
        self.xz_start.setValue(float(xz.get("z_start_m", 0.0)))
        self.xz_stop.setValue(float(xz.get("z_stop_m", 0.02)))
        self.xz_count.setValue(int(xz.get("count", 64)))
        self.source_editor.set_values(self.source_model.config)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("MainToolbar")
        self.addToolBar(toolbar)
        self.new_action = toolbar.addAction("New")
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.open_action = toolbar.addAction("Open")
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.save_action = toolbar.addAction("Save")
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        toolbar.addSeparator()
        self.validate_action = toolbar.addAction("Validate")
        self.run_action = toolbar.addAction("Run")
        self.run_action.setShortcut("F5")
        self.stop_action = toolbar.addAction("Stop")
        self.stop_action.setEnabled(False)

        self.source_editor = PropertyEditor(self)
        source_tab = QWidget(self)
        source_layout = QVBoxLayout(source_tab)
        source_layout.addWidget(self.source_editor)
        self.import_source = QPushButton("Import source field…", source_tab)
        source_layout.addWidget(self.import_source)
        source_layout.addStretch()

        layers_tab = QWidget(self)
        layers_layout = QVBoxLayout(layers_tab)
        self.layer_list = QListView(layers_tab)
        self.layer_list.setModel(self.layer_model)
        self.layer_list.setAlternatingRowColors(True)
        self.layer_list.setSelectionMode(QListView.SelectionMode.SingleSelection)
        layers_layout.addWidget(self.layer_list, 1)
        add_row = QHBoxLayout()
        self.add_kind = QComboBox(layers_tab)
        for kind in ("propagation", "lens", "aperture", "interface", "mask"):
            self.add_kind.addItem(kind.title(), kind)
        self.add_button = QPushButton("Add", layers_tab)
        self.remove_button = QPushButton("Remove", layers_tab)
        self.up_button = QPushButton("↑", layers_tab)
        self.down_button = QPushButton("↓", layers_tab)
        for widget in (self.add_kind, self.add_button, self.remove_button, self.up_button, self.down_button):
            add_row.addWidget(widget)
        layers_layout.addLayout(add_row)
        self.layer_editor = PropertyEditor(layers_tab)
        layers_layout.addWidget(self.layer_editor)
        self.import_mask = QPushButton("Import selected mask…", layers_tab)
        layers_layout.addWidget(self.import_mask)

        settings_tab = QWidget(self)
        settings_layout = QVBoxLayout(settings_tab)
        self.xz_enabled = QCheckBox("Calculate XZ intensity scan", settings_tab)
        self.xz_method = QComboBox(settings_tab)
        for method in ("asm", "blas", "fresnel_tf", "rs_fft"):
            self.xz_method.addItem(method.upper(), method)
        self.xz_start = self._double(0.0)
        self.xz_stop = self._double(0.02)
        self.xz_count = QSpinBox(settings_tab)
        self.xz_count.setRange(2, 1000)
        self.xz_count.setValue(64)
        for label, widget in (("", self.xz_enabled), ("Method", self.xz_method), ("Start z [m]", self.xz_start), ("Stop z [m]", self.xz_stop), ("Samples", self.xz_count)):
            row = QHBoxLayout(); row.addWidget(QLabel(label)); row.addWidget(widget); settings_layout.addLayout(row)
        settings_layout.addStretch()

        left_tabs = QTabWidget(self)
        left_tabs.addTab(source_tab, "Source")
        left_tabs.addTab(layers_tab, "Layers")
        left_tabs.addTab(settings_tab, "XZ")
        self.plots = ResultPlotWidget(self)
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        right_tabs = QTabWidget(self)
        right_tabs.addTab(self.plots, "Results")
        right_tabs.addTab(self.log, "Log")
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_tabs); splitter.addWidget(right_tabs)
        splitter.setStretchFactor(1, 1); splitter.setSizes([390, 890])
        self.setCentralWidget(splitter)
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        self.status = QStatusBar(self)
        self.status.addPermanentWidget(self.progress)
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _connect(self) -> None:
        self.source_editor.valuesChanged.connect(self.source_model.replace)
        self.source_model.changed.connect(lambda values: self.source_editor.set_values(values))
        self.layer_list.selectionModel().currentChanged.connect(self._select_layer)
        self.layer_editor.valuesChanged.connect(self._update_layer)
        self.add_button.clicked.connect(self._add_layer)
        self.remove_button.clicked.connect(self._remove_layer)
        self.up_button.clicked.connect(lambda: self._move_layer(-1))
        self.down_button.clicked.connect(lambda: self._move_layer(1))
        self.import_source.clicked.connect(self._import_source)
        self.import_mask.clicked.connect(self._import_mask)
        self.new_action.triggered.connect(self.new_project)
        self.open_action.triggered.connect(self.open_project)
        self.save_action.triggered.connect(self.save_project)
        self.validate_action.triggered.connect(self.validate_project)
        self.run_action.triggered.connect(self.run_simulation)
        self.stop_action.triggered.connect(self.service.cancel)
        self.service.started.connect(self._run_started)
        self.service.progress.connect(self._progress)
        self.service.result.connect(self._result)
        self.service.error.connect(self._error)
        self.service.cancelled.connect(lambda text: self._append(f"Cancelled: {text}"))
        self.service.finished.connect(self._run_finished)

    def new_project(self) -> None:
        self.project_path = None
        self._project_id = uuid4()
        self._project_name = "Untitled project"
        self._project_compute = None
        self._project_assets = ()
        self._project_metadata = {}
        self.load_dict({"source": DEFAULT_SOURCE, "layers": []})
        self.setWindowTitle("OptiProp Near-field Workbench")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "OptiProp project (*.json *.oprop)")
        if not path: return
        try:
            from optiprop.project import load_project, require_valid_assets

            project = load_project(path)
            require_valid_assets(project.assets, Path(path).parent)
            self.load_dict(project.to_dict(), base_dir=Path(path).parent)
            self.project_path = Path(path)
            self.setWindowTitle(f"{self.project_path.name} — OptiProp")
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def save_project(self) -> None:
        path = self.project_path
        if path is None:
            selected, _ = QFileDialog.getSaveFileName(self, "Save project", "untitled.oprop", "OptiProp project (*.oprop *.json)")
            if not selected: return
            path = Path(selected)
        try:
            from optiprop.project import save_project

            if self._project_name == "Untitled project":
                self._project_name = path.stem
            save_project(self.persisted_project(), path)
            self.project_path = path
            self.status.showMessage(f"Saved {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def validate_project(self) -> bool:
        try:
            field = build_source(self.source_model.config).create()
            report = build_system(self.layer_model.layers).validate(field)
            for issue in report: self._append(f"{issue.severity.value}: {issue.message}")
            self.status.showMessage("Validation passed" if report.is_valid else "Validation has errors", 5000)
            return report.is_valid
        except Exception as exc:
            self._append(f"Validation error: {exc}"); self.status.showMessage("Validation failed", 5000)
            return False

    def run_simulation(self) -> None:
        if self.validate_project(): self.service.start(self.project_dict())

    def _add_layer(self) -> None:
        row = self.layer_model.append_kind(self.add_kind.currentData())
        self.layer_list.setCurrentIndex(self.layer_model.index(row))

    def _remove_layer(self) -> None:
        row = self.layer_list.currentIndex().row()
        if self.layer_model.remove_row(row) and self.layer_model.rowCount():
            self.layer_list.setCurrentIndex(self.layer_model.index(min(row, self.layer_model.rowCount() - 1)))

    def _move_layer(self, delta: int) -> None:
        row = self.layer_list.currentIndex().row()
        target = self.layer_model.move(row, delta)
        if target >= 0: self.layer_list.setCurrentIndex(self.layer_model.index(target))

    def _select_layer(self, current, _previous) -> None:
        self.layer_editor.set_values(self.layer_model.layer(current.row()) if current.isValid() else {})

    def _update_layer(self, values: dict) -> None:
        row = self.layer_list.currentIndex().row()
        if row >= 0: self.layer_model.update_layer(row, values)

    def _import_source(self) -> None:
        dialog = FieldImportDialog(self.source_model.config.get("path", ""), self)
        if dialog.exec(): self.source_model.update({"kind": "imported_field", **dialog.import_config})

    def _import_mask(self) -> None:
        row = self.layer_list.currentIndex().row()
        if row < 0 or self.layer_model.layer(row)["type"] != "mask":
            QMessageBox.information(self, "Select mask", "Select a complex mask layer first."); return
        dialog = FieldImportDialog(self.layer_model.layer(row).get("path", ""), self)
        if dialog.exec(): self.layer_model.update_layer(row, dialog.import_config)

    def _run_started(self) -> None:
        self.run_action.setEnabled(False); self.stop_action.setEnabled(True)
        self.progress.setVisible(True); self.progress.setValue(0)

    def _progress(self, value: int, text: str) -> None:
        self.progress.setValue(value); self.status.showMessage(text); self._append(text)

    def _result(self, output) -> None:
        self.plots.set_output(output)
        suffix = " (cache)" if output.cache_hit else ""
        self._append(f"Completed in {output.result.elapsed_s:.4g} s{suffix}")

    def _error(self, message: str, details: str) -> None:
        self._append(details); self.status.showMessage(f"Error: {message}")

    def _run_finished(self) -> None:
        self.run_action.setEnabled(True); self.stop_action.setEnabled(False); self.progress.setVisible(False)

    def _append(self, text: str) -> None:
        self.log.append(text)

    @staticmethod
    def _double(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox(); box.setDecimals(12); box.setRange(-1e6, 1e6); box.setValue(value); return box

    def _restore_settings(self) -> None:
        geometry = QSettings("OptiProp", "NearFieldWorkbench").value("geometry")
        if geometry is not None: self.restoreGeometry(geometry)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.service.is_running: self.service.cancel()
        QSettings("OptiProp", "NearFieldWorkbench").setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def _resolve_config_path(config: dict, base_dir: Path) -> None:
    value = config.get("path")
    if value and not Path(value).is_absolute():
        config["path"] = str((base_dir / value).resolve())


def _make_paths_portable(config: dict, base_dir: Path) -> None:
    value = config.get("path")
    if not value:
        return
    try:
        config["path"] = Path(value).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        # External fields stay absolute and are intentionally not claimed as
        # portable project assets.
        config["path"] = str(Path(value).resolve())


__all__ = ["MainWindow"]
