"""Local execution boundary: validated projects, immutable runs, bounded worker."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
from threading import RLock
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.io import savemat
import torch

from ..io import load_field, save_field, save_zbf
from ..io.mat import canonical_mat_payload
from ..project import OptiPropProject, project_to_domain, require_valid_assets
from ..project.assets import AssetReference
from ..propagation import CancellationToken, PaddingSpec, PropagationSpec, propagate_angular_spectrum
from ..system import IncidentSource, OpticalSystem

MAX_PIXELS = 1024 * 1024
MAX_WORK_BYTES = 512 * 1024**2
_PLOT_LOCK = RLock()


def _json_write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _grid_limit(config):
    if config:
        nx, ny = config.get("nx"), config.get("ny")
        if not isinstance(nx, int) or not isinstance(ny, int) or min(nx, ny) < 2 or nx * ny > MAX_PIXELS:
            raise ValueError("Workbench grids need Nx, Ny >= 2 and at most 1,048,576 pixels.")


def build_project(document, asset_root):
    """Reconstruct canonical API objects without permitting unbounded web jobs."""
    project = OptiPropProject.from_dict(document)
    configs = document["optical_system"].get("layers", [])
    if len(configs) > 32:
        raise ValueError("Workbench supports at most 32 optical steps per run.")
    _grid_limit(document["source"].get("grid"))
    for layer in configs:
        if layer.get("type") == "Binary2LensLayer" and len(layer.get("coefficients", [])) > 32:
            raise ValueError("At most 32 Binary2 coefficients are supported by the workbench.")
        spec = layer.get("spec", {})
        _grid_limit(spec.get("output_grid"))
        padding = spec.get("padding", {})
        if padding.get("factor") is not None and not 1 <= float(padding["factor"]) <= 3:
            raise ValueError("Workbench padding factor must be between 1 and 3.")
        shape = padding.get("shape")
        if shape and (len(shape) != 2 or min(shape) < 2 or math.prod(shape) > 4 * MAX_PIXELS):
            raise ValueError("Padding shape exceeds workbench limits.")
    require_valid_assets(project.assets, asset_root)
    source, system = project_to_domain(project, asset_loader=lambda asset: load_field(asset.resolve(asset_root)))
    precision = project.compute_settings.precision
    if precision not in ("complex64", "complex128"):
        raise ValueError("Choose explicit complex64 or complex128 precision.")
    device = project.compute_settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose CPU. Windows portable builds use CPU PyTorch.")
    if source.grid is not None:
        _grid_limit({"nx": source.grid.nx, "ny": source.grid.ny})
    field = source.create(dtype=getattr(torch, precision), device=device)
    pixels = field.grid.nx * field.grid.ny
    snapshot_bytes = field.data.numel() * field.data.element_size() * (len(configs) + 2)
    if pixels > MAX_PIXELS or snapshot_bytes > MAX_WORK_BYTES:
        raise ValueError("Run exceeds workbench snapshot memory limit; reduce grid or number of steps.")
    for layer in configs:
        spec = layer.get("spec", {})
        padding = spec.get("padding", {})
        factor = float(padding.get("factor") or (1 if padding.get("mode") == "none" else 2))
        if pixels * factor**2 > 4 * MAX_PIXELS:
            raise ValueError("Padded grid exceeds workbench limit; reduce grid or padding.")
    return project, system, field


def issues_json(report):
    return [{"severity": issue.severity.value, "message": issue.message,
             "code": issue.code, "layer_id": issue.layer_id} for issue in report]


def field_info(field):
    return {"nx": field.grid.nx, "ny": field.grid.ny, "dx_m": field.grid.dx,
            "dy_m": field.grid.dy, "z_m": field.z_m, "wavelength_m": field.wavelength_m,
            "medium_index": {"real": field.medium_index.real, "imag": field.medium_index.imag},
            "x_center_m": field.grid.x_center, "y_center_m": field.grid.y_center,
            "components": list(field.components), "dtype": str(field.dtype)}


def _bar(fig, ax, artist, label):
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=.12)
    bar = fig.colorbar(artist, cax=cax)
    bar.set_label(label, fontsize=14)
    bar.ax.tick_params(labelsize=12)


def field_png(field, quantity="amplitude_phase"):
    """Render actual source fields, never the lens transmission in their place."""
    if quantity not in ("amplitude_phase", "intensity"):
        raise ValueError("Unknown field plot.")
    with _PLOT_LOCK:
        data = field.data.detach().cpu().numpy()
        ncols = len(field.components)
        rows = 2 if quantity == "amplitude_phase" else 1
        fig = Figure(figsize=(6.8 * ncols, 5.8 * rows), dpi=110)
        FigureCanvasAgg(fig)
        axes = fig.subplots(rows, ncols, squeeze=False)
        fig.subplots_adjust(left=.075, right=.92, top=.88, bottom=.10, wspace=.6, hspace=.4)
        amp_max = max(float(np.abs(data).max()), np.finfo(float).tiny)
        extent = np.asarray(field.grid.pixel_extent) * 1e6
        for col, component in enumerate(field.components):
            intensity = (np.abs(data[col]) / amp_max)**2
            for row in range(rows):
                ax = axes[row, col]
                if quantity == "intensity":
                    shown, cmap, vmin, vmax, label = intensity, "magma", 0, 1, "I / shared max"
                elif row == 0:
                    shown, cmap, vmin, vmax, label = np.abs(data[col]) / amp_max, "viridis", 0, 1, "Amplitude / shared max"
                else:
                    shown = np.ma.masked_where(intensity <= .001 * intensity.max(), np.angle(data[col]))
                    cmap, vmin, vmax, label = "twilight", -np.pi, np.pi, "Phase (rad)"
                artist = ax.imshow(shown, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(component + " — " + label.split(" /")[0], fontsize=18)
                ax.set_xlabel("x (µm)", fontsize=16); ax.set_ylabel("y (µm)", fontsize=16)
                ax.tick_params(labelsize=13)
                _bar(fig, ax, artist, label)
        fig.suptitle(f"Field at z = {field.z_m * 1e6:g} µm · λ = {field.wavelength_m * 1e9:g} nm\n"
                     "Phase masked below 0.1% of component peak intensity", fontsize=17)
        stream = BytesIO(); fig.savefig(stream, format="png", bbox_inches="tight")
        return stream.getvalue()


def xz_png(field, xz, z):
    with _PLOT_LOCK:
        fig = Figure(figsize=(6.8 * len(field.components), 7), dpi=110)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, len(field.components), squeeze=False)[0]
        fig.subplots_adjust(left=.08, right=.92, bottom=.12, top=.85, wspace=.6)
        shared_max = max(float(xz.max()), np.finfo(float).tiny)
        x = field.grid.x_coordinates().numpy() * 1e6
        y_slice = float(field.grid.y_coordinates()[field.grid.ny // 2]) * 1e6
        for c, ax in enumerate(axes):
            artist = ax.pcolormesh(x, z * 1e6, np.maximum(xz[c] / shared_max, 1e-5),
                                   shading="auto", cmap="magma", norm=LogNorm(1e-4, 1))
            ax.set_title(field.components[c], fontsize=18)
            ax.set_xlabel("x (µm)", fontsize=16); ax.set_ylabel("absolute z (µm)", fontsize=16)
            ax.tick_params(labelsize=13); _bar(fig, ax, artist, "I / shared max (log)")
        fig.suptitle(f"ASM XZ · y = {y_slice:g} µm · homogeneous n = {field.medium_index.real:g}", fontsize=17)
        stream = BytesIO(); fig.savefig(stream, format="png", bbox_inches="tight")
        return stream.getvalue()


class WorkbenchService:
    """One cancellable numerical task at a time, with immutable saved settings."""

    def __init__(self, output_root):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optiprop")
        self.jobs = {}
        self.active = None

    def validate(self, document, asset_root):
        _, system, field = build_project(document, asset_root)
        report = system.validate(field)
        return {"valid": report.is_valid, "issues": issues_json(report), "field": field_info(field)}

    def start(self, payload, kind="run"):
        # Snapshot JSON input before it can be changed by the UI.
        payload = json.loads(json.dumps(payload, allow_nan=False))
        with self.lock:
            if self.active is not None:
                raise ValueError("A task is running. Wait for it or cancel it first.")
            identifier = uuid4().hex
            self.jobs[identifier] = {"id": identifier, "kind": kind, "status": "queued", "message": "Queued",
                                     "token": CancellationToken(), "payload": payload}
            self.active = identifier
            # Keep at most two previous runs in RAM; on-disk output is not deleted.
            old = [key for key in self.jobs if key != identifier and key != payload.get("job")]
            for key in old[:-2]:
                del self.jobs[key]
            self.executor.submit(self._execute, identifier)
            return {"id": identifier}

    def status(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            return {key: job[key] for key in ("id", "kind", "status", "message", "output_dir", "issues", "project", "field") if key in job}

    def cancel(self, identifier):
        with self.lock:
            self.jobs[identifier]["token"].cancel("Cancelled by user")
        return {"cancel_requested": True}

    def _execute(self, identifier):
        job = self.jobs[identifier]
        token, payload = job["token"], job["payload"]
        try:
            with self.lock:
                job.update(status="running", message="Building project")
            directory = self.output_root / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + identifier[:8])
            directory.mkdir(exist_ok=False)
            with self.lock:
                job["output_dir"] = str(directory)
            token.throw_if_cancelled()
            if job["kind"] == "run":
                project, system, field = build_project(payload["project"], payload.get("asset_root", "."))
                _json_write(directory / "project.json", project.to_dict())
                token.throw_if_cancelled()
                with self.lock:
                    job.update(message="Propagating optical system", project=project.to_dict())
                with torch.no_grad():
                    result = system.execute(field, cancel_token=token)
                if not torch.isfinite(result.output_field.data).all():
                    raise ValueError("Non-finite optical field; review sampling and parameters.")
                token.throw_if_cancelled()
                with self.lock:
                    job.update(result=result, source_id=str(project.source["id"]),
                               field=field_info(result.output_field), issues=issues_json(result.validation))
            elif job["kind"] == "xz":
                field = self.observed(payload)
                count = payload.get("count", 81)
                stop = float(payload.get("distance_m", 100e-6))
                if isinstance(count, bool) or not isinstance(count, int) or not 3 <= count <= 201 or not math.isfinite(stop) or stop <= 0:
                    raise ValueError("XZ needs positive distance and 3–201 samples.")
                z = np.linspace(0, stop, count)
                rows = []
                with torch.no_grad():
                    for i, distance in enumerate(z):
                        token.throw_if_cancelled()
                        out = field if distance == 0 else propagate_angular_spectrum(field, PropagationSpec(
                            "asm", float(distance), medium_index=field.medium_index, padding=PaddingSpec.by_factor(2)), cancel_token=token).field
                        if not torch.isfinite(out.data).all():
                            raise ValueError("XZ contains non-finite optical fields.")
                        rows.append(out.data.abs().square()[:, field.grid.ny // 2, :].cpu().numpy())
                        with self.lock:
                            job["message"] = f"XZ {i+1}/{count}"
                xz = np.stack(rows, axis=1)
                absolute_z = z + field.z_m
                png = xz_png(field, xz, absolute_z)
                (directory / "xz_intensity.png").write_bytes(png)
                np.savez_compressed(directory / "xz.npz", intensity=xz, z_m=absolute_z,
                                    x_m=field.grid.x_coordinates().numpy(), y_slice_m=float(field.grid.y_coordinates()[field.grid.ny // 2]))
                _json_write(directory / "observation.json", {"parent_run": payload["job"], "selection": payload, "field": field_info(field)})
                with self.lock:
                    job["png"] = png
            else:
                raise ValueError("Unknown job kind")
            token.throw_if_cancelled()
            with self.lock:
                job.update(status="completed", message="Complete")
        except Exception as exc:
            from ..propagation import PropagationCancelled
            with self.lock:
                job.update(status="cancelled" if isinstance(exc, PropagationCancelled) else "failed", message=str(exc))
        finally:
            with self.lock:
                self.active = None
                if "output_dir" in job:
                    _json_write(Path(job["output_dir"]) / "run.json", self.status(identifier))

    def observed(self, payload):
        with self.lock:
            job = self.jobs[payload["job"]]
            if job["kind"] != "run" or job["status"] != "completed":
                raise ValueError("Select a completed optical run.")
            result = job["result"]
            identifier, side = payload.get("layer_id", job["source_id"]), payload.get("side", "after")
            if side not in ("before", "after"):
                raise ValueError("side must be before or after")
            if identifier == job["source_id"]:
                return result.input_field
            layer = result.result_for(identifier)
            return layer.input_field if side == "before" else layer.output_field

    def view(self, payload):
        field = self.observed(payload)
        return {"field": field_info(field), "png": base64.b64encode(field_png(field, payload.get("quantity", "amplitude_phase"))).decode("ascii")}

    def export(self, payload):
        field = self.observed(payload)
        kind = payload.get("format", "npz")
        if kind not in ("npz", "mat", "zbf", "png"):
            raise ValueError("Choose NPZ, MAT, ZBF or PNG.")
        directory = Path(self.jobs[payload["job"]]["output_dir"])
        path = directory / ("field-" + uuid4().hex[:12] + "." + kind)
        field = field.with_data(field.data.to(torch.complex128))
        if kind == "zbf":
            if any(n & (n-1) for n in (field.grid.nx, field.grid.ny)):
                raise ValueError("ZBF requires Nx and Ny to be powers of two. Change sampling explicitly; no silent padding.")
            save_zbf(field, path, length_unit="mm")
        elif kind == "mat":
            # Interoperable Ex/Ey MATLAB layout requested by the user, plus SI metadata.
            values = canonical_mat_payload(field)
            values.update({name.upper(): field.data[i].detach().cpu().numpy() for i, name in enumerate(field.components)})
            values.update(dx=field.grid.dx, dy=field.grid.dy, Nx=field.grid.nx, Ny=field.grid.ny,
                          wavelength=field.wavelength_m, n=field.medium_index, z=field.z_m)
            savemat(path, values)
        elif kind == "png":
            path.write_bytes(field_png(field, payload.get("quantity", "amplitude_phase")))
        else:
            save_field(field, path)
        _json_write(path.with_suffix(path.suffix + ".json"), {"run_id": payload["job"], "selection": payload,
                    "field": field_info(field), "file_length_unit": "mm" if kind == "zbf" else "m",
                    "note": "ZBF omits absolute z and grid center; coordinates are retained in this sidecar." if kind == "zbf" else ""})
        return {"path": str(path)}

    def import_source(self, path):
        path = Path(path).expanduser().resolve()
        if path.stat().st_size > 512 * 1024**2:
            raise ValueError("Workbench imports are limited to 512 MiB per field file.")
        field = load_field(path)
        _grid_limit({"nx": field.grid.nx, "ny": field.grid.ny})
        # Copy into a canonical managed asset; the input file is never modified.
        directory = self.output_root / ("import-" + uuid4().hex[:12])
        directory.mkdir()
        save_field(field, directory / "source.npz")
        asset = AssetReference.from_file(directory / "source.npz", project_directory=directory)
        project = OptiPropProject.from_domain(IncidentSource.from_field(field), OpticalSystem(()),
                                             name=path.stem, source_asset_id=asset.id, assets=(asset,))
        document = project.to_dict()
        document["compute_settings"].update(device="cpu", precision="complex128")
        _json_write(directory / "project.json", document)
        return {"project": document, "asset_root": str(directory)}

    def close(self):
        with self.lock:
            if self.active:
                self.jobs[self.active]["token"].cancel("Workbench shutting down")
        self.executor.shutdown(wait=True, cancel_futures=True)
