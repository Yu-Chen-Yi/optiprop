"""Load a metalens project JSON and calculate focal fields and XZ maps.

Run from any working directory with the repository's Python dependencies:
    python Example/Metalens_Focus_Project.py
    python Example/Metalens_Focus_Project.py --phase-model binary2

Outputs are saved below output/metalens_focus/<timestamp>/, never over an old run.
These are scalar phase demonstrations, not database/full-wave metalenses.
Binary2 uses four Taylor terms of the ideal profile, not an exact conversion.
The JSON defines the optical system. The example's axial diagnostic scans from
the metalens output, independently for each z, without changing the saved path.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optiprop.project import load_project, project_to_domain
from optiprop.propagation import PaddingSpec, propagate_angular_spectrum
from optiprop.system import Binary2LensLayer, IdealLensLayer, PropagationLayer


def fwhm(x: np.ndarray, intensity: np.ndarray) -> float:
    """Interpolated central-peak width, without including disconnected sidelobes."""
    peak = int(np.argmax(intensity))
    half = intensity[peak] / 2
    left = peak
    right = peak
    while left > 0 and intensity[left] >= half:
        left -= 1
    while right < len(x) - 1 and intensity[right] >= half:
        right += 1
    if intensity[left] >= half or intensity[right] >= half:
        raise ValueError("Half-maximum crossing lies outside the observation window.")
    x_left = np.interp(half, intensity[left:left + 2], x[left:left + 2])
    x_right = np.interp(half, intensity[right - 1:right + 1][::-1], x[right - 1:right + 1][::-1])
    return float(x_right - x_left)


def colorbar(fig, ax, artist, label):
    """Keep each colorbar exactly aligned to its parent axes height."""
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.12)
    bar = fig.colorbar(artist, cax=cax)
    bar.set_label(label)
    return bar


def save_figures(output, post_lens, focal, xz, z_values, model_label):
    x = focal.grid.x_coordinates().numpy() * 1e6
    y = focal.grid.y_coordinates().numpy() * 1e6
    extent = np.asarray(focal.grid.pixel_extent) * 1e6
    field = focal.data.detach().cpu().numpy()
    intensity = np.abs(field) ** 2
    amplitude_max = float(np.abs(field).max())
    intensity_max = float(intensity.max())
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 19,
                         "axes.labelsize": 17, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "savefig.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(left=.075, right=.91, bottom=.13, top=.85, wspace=.6)
    for c, ax in enumerate(axes):
        shown = ax.imshow(intensity[c] / intensity_max, extent=extent, origin="lower",
                          cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set(title=f"{focal.components[c]} focal intensity", xlabel="x (µm)",
               ylabel="y (µm)", xlim=(-8, 8), ylim=(-8, 8))
        colorbar(fig, ax, shown, "I / max(I_Ex, I_Ey)")
    fig.suptitle(f"{model_label} metalens: reference focal plane z = {focal.z_m * 1e6:g} µm")
    fig.savefig(output / "focus_intensity.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.subplots_adjust(left=.075, right=.91, bottom=.08, top=.89, wspace=.6, hspace=.4)
    for c in range(2):
        for row in range(2):
            ax = axes[row, c]
            if row == 0:
                data = np.abs(field[c]) / amplitude_max
                cmap, vmin, vmax, label = "viridis", 0, 1, "Amplitude / global max"
            else:
                data = np.ma.masked_where(intensity[c] < .001 * intensity[c].max(), np.angle(field[c]))
                cmap, vmin, vmax, label = "twilight", -np.pi, np.pi, "Phase (rad)"
            shown = ax.imshow(data, extent=extent, origin="lower", cmap=cmap,
                              vmin=vmin, vmax=vmax, aspect="equal")
            ax.set(title=f"{focal.components[c]} — {'amplitude' if row == 0 else 'phase'}",
                   xlabel="x (µm)", ylabel="y (µm)", xlim=(-8, 8), ylim=(-8, 8))
            colorbar(fig, ax, shown, label)
    fig.suptitle(f"{model_label}: focal field (phase masked below 0.1% intensity)")
    fig.savefig(output / "focal_amplitude_phase.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    fig.subplots_adjust(left=.075, right=.91, bottom=.12, top=.85, wspace=.6)
    scale = float(xz.max())
    slice_y = y[focal.grid.ny // 2]
    for c, ax in enumerate(axes):
        shown = ax.pcolormesh(x, (z_values + post_lens.z_m) * 1e6,
                              np.maximum(xz[c] / scale, 1e-5), shading="auto",
                              cmap="magma", norm=LogNorm(vmin=1e-4, vmax=1))
        ax.axhline(focal.z_m * 1e6, color="white", linestyle="--", linewidth=1)
        ax.set(title=f"{focal.components[c]} XZ intensity", xlabel="x (µm)",
               ylabel="z (µm)", xlim=(-35, 35))
        colorbar(fig, ax, shown, "I / shared maximum (log)")
    fig.suptitle(f"{model_label} · ASM · y = {slice_y:g} µm · dashed: reference focus")
    fig.savefig(output / "xz_intensity.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-points", type=int, default=121)
    parser.add_argument("--phase-model", choices=("ideal", "binary2"), default="ideal")
    args = parser.parse_args()
    if args.scan_points < 3:
        parser.error("--scan-points must be at least 3")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    project_path = REPO_ROOT / f"Example/projects/metalens_focus_{args.phase_model}.json"
    project = load_project(project_path)
    source, system = project_to_domain(project)
    if (len(system.layers) != 2 or not isinstance(system.layers[0], (IdealLensLayer, Binary2LensLayer))
            or not isinstance(system.layers[1], PropagationLayer)):
        raise ValueError("This focal diagnostic expects one lens followed by one propagation segment.")
    lens, segment = system.layers
    reference_focal_length = (lens.focal_length_m if isinstance(lens, IdealLensLayer)
                              else float(project.metadata["reference_focal_length_m"]))
    model_label = "Ideal" if isinstance(lens, IdealLensLayer) else "Binary2"
    if not np.isfinite(reference_focal_length) or reference_focal_length <= 0 or segment.spec.distance_m <= 0:
        raise ValueError("The focusing example requires positive focal length and propagation distance.")
    output = REPO_ROOT / "output/metalens_focus" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        input_field = source.create(dtype=torch.complex128, device="cpu")
        result = system.execute(input_field)
        post_lens = result.result_for(lens.id).output_field
        focal = result.output_field
        if not torch.isfinite(focal.data).all():
            raise RuntimeError("Focal field contains non-finite values.")
        z_values = np.linspace(0, 1.5 * reference_focal_length, args.scan_points)
        iy, ix = focal.grid.ny // 2, focal.grid.nx // 2
        rows, central_intensity = [], []
        for i, distance in enumerate(z_values):
            propagated = post_lens if distance == 0 else propagate_angular_spectrum(
                post_lens, replace(segment.spec, distance_m=float(distance))).field
            intensities = propagated.data.abs().square()
            rows.append(intensities[:, iy, :].cpu().numpy())
            # Even grids have no sample exactly at zero; report the explicit slice.
            central_intensity.append(float(intensities[:, iy, ix].sum()))
            if i % 30 == 0:
                print(f"XZ scan {i + 1}/{len(z_values)}", flush=True)
        xz = np.stack(rows, axis=1)
        focal_i = focal.data.abs().square().cpu().numpy()
        x = focal.grid.x_coordinates().numpy()
        y = focal.grid.y_coordinates().numpy()
        total = focal_i.sum(axis=0)
        widths = [fwhm(x, item[iy]) * 1e6 for item in focal_i]
        maximum_y, maximum_x = np.unravel_index(total.argmax(), total.shape)
        larger_padding = propagate_angular_spectrum(
            post_lens, replace(segment.spec, padding=PaddingSpec.by_factor(3))).field
        check_peak = float(larger_padding.data.abs().square().sum(dim=0).max())
        peak = float(total.max())
        padding_error = abs(peak - check_peak) / check_peak
        best_z = float(z_values[int(np.argmax(central_intensity))])
        # Regressions should fail loudly; these broad checks are not full convergence proof.
        if not .75 * reference_focal_length <= best_z <= 1.25 * reference_focal_length:
            raise RuntimeError("Axial maximum is unexpectedly far from the designed focus.")
        if abs(x[maximum_x]) > 2 * focal.grid.dx or abs(y[maximum_y]) > 2 * focal.grid.dy:
            raise RuntimeError("The on-axis lens produced an off-center maximum.")
        if padding_error > .02:
            raise RuntimeError("Focal peak differs by over 2% between padding factors 2 and 3.")
        input_i = input_field.data.abs().square().sum(dim=0)
        report = {
            "project": project.to_dict(),
            "project_sha256": hashlib.sha256(project_path.read_bytes()).hexdigest(),
            "model": f"{model_label} scalar phase, not database/full-wave",
            "dtype": str(focal.dtype),
            "design_focal_length_um": reference_focal_length * 1e6,
            "evaluated_plane_z_um": focal.z_m * 1e6,
            "scanned_axial_peak_after_lens_um": best_z * 1e6,
            "scan_step_um": float(z_values[1] - z_values[0]) * 1e6,
            "slice_y_um": float(y[iy]) * 1e6,
            "axial_sample_x_um": float(x[ix]) * 1e6,
            "focal_fwhm_x_um": dict(zip(focal.components, widths)),
            "peak_xy_um": [float(x[maximum_x]) * 1e6, float(y[maximum_y]) * 1e6],
            "peak_vs_incident_intensity": peak / float(input_i.max()),
            "focal_peak_relative_change_padding_2_vs_3": padding_error,
            "validation": "finite field, near-design axial peak, centered peak, padding spot-check; not a full sampling convergence study",
        }
        save_figures(output, post_lens, focal, xz, z_values, model_label)
        # Compact diagnostics only; no large 3D field volume is retained.
        np.savez_compressed(output / "focus_scan.npz", x_m=x, z_after_lens_m=z_values,
                            xz_intensity=xz, axial_intensity=central_intensity,
                            focal_center_profiles=focal_i[:, iy, :], slice_y_m=y[iy])
        (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "project"}, indent=2))
        print(f"Output: {output}")


if __name__ == "__main__":
    main()
