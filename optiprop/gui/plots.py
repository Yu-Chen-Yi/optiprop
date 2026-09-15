"""Matplotlib Qt result views for coherent-field diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QComboBox, QTabWidget, QVBoxLayout, QWidget


class _Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

    def image(self, data, extent, title: str, colorbar_label: str, cmap="viridis") -> None:
        self.axes.clear()
        shown = self.axes.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        self.axes.set(title=title, xlabel="x [mm]", ylabel="y [mm]")
        # Recreating the figure is unnecessary; remove the previous colorbar axes.
        while len(self.figure.axes) > 1:
            self.figure.delaxes(self.figure.axes[-1])
        colorbar = self.figure.colorbar(shown, ax=self.axes)
        colorbar.set_label(colorbar_label)
        self.draw_idle()


class ResultPlotWidget(QWidget):
    """Switch between layer snapshots and amplitude/phase/intensity/profile/XZ."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.selector = QComboBox(self)
        self.selector.currentIndexChanged.connect(self._draw_selected)
        self.tabs = QTabWidget(self)
        self.amplitude = _Canvas(self)
        self.phase = _Canvas(self)
        self.intensity = _Canvas(self)
        self.profile = _Canvas(self)
        self.xz = _Canvas(self)
        for label, canvas in (
            ("Amplitude", self.amplitude), ("Phase", self.phase),
            ("Intensity", self.intensity), ("Center profile", self.profile),
            ("XZ scan", self.xz),
        ):
            self.tabs.addTab(canvas, label)
        layout = QVBoxLayout(self)
        layout.addWidget(self.selector)
        layout.addWidget(self.tabs, 1)
        self._output = None

    def set_output(self, output) -> None:
        self._output = output
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItem("Input", output.result.input_field)
        for layer in output.result.layer_results:
            self.selector.addItem(layer.layer_name, layer.output_field)
        self.selector.blockSignals(False)
        self.selector.setCurrentIndex(self.selector.count() - 1)
        self._draw_selected()
        if output.xz_intensity is not None:
            extent = [
                output.x_axis_m[0] * 1e3, output.x_axis_m[-1] * 1e3,
                output.z_axis_m[0] * 1e3, output.z_axis_m[-1] * 1e3,
            ]
            self.xz.image(output.xz_intensity, extent, "XZ intensity", "relative intensity", "magma")
            self.tabs.setTabEnabled(4, True)
        else:
            self.xz.axes.clear()
            self.xz.axes.text(0.5, 0.5, "Enable XZ scan before Run", ha="center", va="center")
            self.xz.draw_idle()
            self.tabs.setTabEnabled(4, False)

    def _draw_selected(self, *_args) -> None:
        field = self.selector.currentData()
        if field is None:
            return
        data = field.data[0].detach().cpu().numpy()
        amplitude = np.abs(data)
        intensity = field.intensity().detach().cpu().numpy()
        phase = np.angle(data)
        xmin, xmax, ymin, ymax = field.grid.pixel_extent
        extent = [xmin * 1e3, xmax * 1e3, ymin * 1e3, ymax * 1e3]
        self.amplitude.image(amplitude, extent, "Amplitude", "amplitude")
        self.phase.image(phase, extent, "Wrapped phase", "radians", "twilight")
        self.intensity.image(intensity, extent, "Intensity", "relative intensity", "magma")
        x = field.grid.x_coordinates().cpu().numpy() * 1e3
        self.profile.axes.clear()
        self.profile.axes.plot(x, intensity[intensity.shape[0] // 2])
        self.profile.axes.set(title="Center-line intensity", xlabel="x [mm]", ylabel="relative intensity")
        self.profile.axes.grid(True, alpha=0.25)
        self.profile.draw_idle()


__all__ = ["ResultPlotWidget"]
