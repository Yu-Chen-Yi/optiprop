"""ZEMAX Beam File (ZBF) reader / writer / plotter.

File layout (little-endian):
  header  : 9 x int32   -> [version, nx, ny, ispol, unit, 0, 0, 0, 0]
  header2 : 20 x float64 (version 1) or 11 x float64 (version 0)
  Ex data : nx*ny*2 x float64, interleaved (re, im), x index varies fastest
  Ey data : same as Ex, present only when ispol == 1

Field arrays here use shape (ny, nx): E[iy, ix], row = y, column = x.
This matches the Zemax storage order (x fastest) and makes read/write exact
inverses. Note: the reference MATLAB read_zbf.m returns the transpose of
this (it assumes y varies fastest), which only agrees on symmetric square
grids.

Usage:
    from zbf import read_zbf, write_zbf, plot_beam
    beam = read_zbf("beam.zbf")
    plot_beam(beam, "output_folder")

    # or from the command line:
    python zbf.py beam.zbf -o output_folder
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UNIT_NAMES = {0: "mm", 1: "cm", 2: "in", 3: "m"}


@dataclass
class PilotRays:
    posX: float = 0.0
    posY: float = 0.0
    rayleighX: float = 0.0
    rayleighY: float = 0.0
    waistX: float = 0.0
    waistY: float = 0.0


@dataclass
class ZbfBeam:
    nx: int
    ny: int
    dx: float
    dy: float
    ispol: int
    lam: float                    # wavelength, in lens units
    Ex: np.ndarray                # complex, shape (ny, nx)
    Ey: np.ndarray | None = None  # complex, shape (ny, nx) when polarized
    pilot: PilotRays = field(default_factory=PilotRays)
    unit: int = 0                 # 0=mm 1=cm 2=in 3=m
    index: float = 1.0            # refractive index in current medium
    version: int = 1

    @property
    def unit_name(self) -> str:
        return UNIT_NAMES.get(self.unit, f"unit#{self.unit}")

    @property
    def x(self) -> np.ndarray:
        """x sample coordinates, grid centered on 0."""
        return (np.arange(self.nx) - self.nx / 2) * self.dx

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.ny) - self.ny / 2) * self.dy

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) for matplotlib imshow."""
        return (-self.nx / 2 * self.dx, self.nx / 2 * self.dx,
                -self.ny / 2 * self.dy, self.ny / 2 * self.dy)

    def describe(self) -> str:
        u = self.unit_name
        lines = [
            f"grid      : nx x ny = {self.nx} x {self.ny}",
            f"spacing   : dx = {self.dx:g} {u}, dy = {self.dy:g} {u}"
            f"  (width {self.nx * self.dx:g} x {self.ny * self.dy:g} {u})",
            f"wavelength: {self.lam:g} {u}",
            f"polarized : {'yes (Ex + Ey)' if self.ispol == 1 else 'no (Ex only)'}",
            f"index     : {self.index:g}",
            f"pilot X   : pos {self.pilot.posX:g}, waist {self.pilot.waistX:g},"
            f" rayleigh {self.pilot.rayleighX:g}",
            f"pilot Y   : pos {self.pilot.posY:g}, waist {self.pilot.waistY:g},"
            f" rayleigh {self.pilot.rayleighY:g}",
        ]
        return "\n".join(lines)


def _read_field(f, nx: int, ny: int) -> np.ndarray:
    raw = np.fromfile(f, dtype="<f8", count=2 * nx * ny)
    if raw.size != 2 * nx * ny:
        raise ValueError("File ended before the full E-field data was read")
    c = raw[0::2] + 1j * raw[1::2]
    return c.reshape(ny, nx)  # x varies fastest in the file -> columns


def read_zbf(filename) -> ZbfBeam:
    """Read a ZEMAX beam file. Returns a ZbfBeam with Ex/Ey of shape (ny, nx)."""
    with open(filename, "rb") as f:
        head = np.fromfile(f, dtype="<i4", count=9)
        if head.size != 9:
            raise ValueError(f"{filename}: too short to be a ZBF file")
        version, nx, ny, ispol, unit = (int(v) for v in head[:5])

        if version == 1:  # current format
            d = np.fromfile(f, dtype="<f8", count=20)
            dx, dy, lam, index = d[0], d[1], d[8], d[9]
            pilot = PilotRays(posX=d[2], rayleighX=d[3], waistX=d[4],
                              posY=d[5], rayleighY=d[6], waistY=d[7])
        else:  # old format
            d = np.fromfile(f, dtype="<f8", count=11)
            dx, dy, lam, index = d[0], d[1], d[4], 1.0
            pilot = PilotRays(posX=d[2], posY=d[2],
                              rayleighX=d[3], rayleighY=d[3],
                              waistX=d[5], waistY=d[5])

        Ex = _read_field(f, nx, ny)
        Ey = _read_field(f, nx, ny) if ispol == 1 else None

    return ZbfBeam(nx=nx, ny=ny, dx=float(dx), dy=float(dy), ispol=ispol,
                   lam=float(lam), Ex=Ex, Ey=Ey, pilot=pilot,
                   unit=unit, index=float(index), version=version)


def write_zbf(beam: ZbfBeam, filename) -> None:
    """Write a ZbfBeam to a version-1 ZBF file (exact inverse of read_zbf)."""
    head = np.zeros(9, dtype="<i4")
    head[0] = 1
    head[1] = beam.nx
    head[2] = beam.ny
    head[3] = beam.ispol
    head[4] = beam.unit

    d = np.zeros(20, dtype="<f8")
    d[0], d[1] = beam.dx, beam.dy
    d[2], d[3], d[4] = beam.pilot.posX, beam.pilot.rayleighX, beam.pilot.waistX
    d[5], d[6], d[7] = beam.pilot.posY, beam.pilot.rayleighY, beam.pilot.waistY
    d[8] = beam.lam
    d[9] = beam.index

    def interleave(E):
        E = np.asarray(E, dtype=complex)
        if E.shape != (beam.ny, beam.nx):
            raise ValueError(f"field shape {E.shape} != (ny, nx) = "
                             f"({beam.ny}, {beam.nx})")
        out = np.empty(2 * beam.nx * beam.ny, dtype="<f8")
        flat = E.ravel()  # C order: x fastest, matching the ZBF layout
        out[0::2] = flat.real
        out[1::2] = flat.imag
        return out

    with open(filename, "wb") as f:
        head.tofile(f)
        d.tofile(f)
        interleave(beam.Ex).tofile(f)
        if beam.ispol == 1:
            if beam.Ey is None:
                raise ValueError("ispol == 1 but beam.Ey is None")
            interleave(beam.Ey).tofile(f)


def crop_center(beam: ZbfBeam, factor: float) -> ZbfBeam:
    """Return a copy cropped to the central 1/factor of the grid (factor > 1)."""
    if factor <= 1:
        return beam
    nx2 = max(2, int(round(beam.nx / factor)))
    ny2 = max(2, int(round(beam.ny / factor)))
    x0 = beam.nx // 2 - nx2 // 2  # keep the grid center at the same sample
    y0 = beam.ny // 2 - ny2 // 2

    def crop(E):
        return None if E is None else E[y0:y0 + ny2, x0:x0 + nx2]

    return ZbfBeam(nx=nx2, ny=ny2, dx=beam.dx, dy=beam.dy, ispol=beam.ispol,
                   lam=beam.lam, Ex=crop(beam.Ex), Ey=crop(beam.Ey),
                   pilot=beam.pilot, unit=beam.unit, index=beam.index,
                   version=beam.version)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_map(fig_ax, data, beam: ZbfBeam, title, cbar_label, cmap,
              vmin=None, vmax=None, show_xlabel=True, show_ylabel=True):
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = fig_ax
    im = ax.imshow(data, origin="lower", extent=beam.extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(f"x ({beam.unit_name})")
    if show_ylabel:
        ax.set_ylabel(f"y ({beam.unit_name})")
    # few ticks so the larger slide fonts never collide
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    # colorbar on a divider axes: always exactly as tall as the image axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    if cmap == "twilight":
        cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    return im


def plot_beam(beam: ZbfBeam, outdir, dpi: int = 200,
              amp_cmap: str = "magma", fontsize: float = 16) -> list[Path]:
    """Save amplitude / phase / intensity maps of Ex (and Ey if polarized).

    Writes <name>_amplitude.png, <name>_phase.png, <name>_intensity.png per
    component plus a combined summary.png into `outdir`. Returns the paths.
    `fontsize` is the base size (ticks); labels and titles scale from it.
    """
    import matplotlib.pyplot as plt

    rc = {
        "font.size": fontsize,
        "axes.titlesize": fontsize * 1.3,
        "axes.labelsize": fontsize * 1.15,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
    with plt.rc_context(rc):
        return _plot_beam_impl(beam, outdir, dpi, amp_cmap, plt)


def _plot_beam_impl(beam: ZbfBeam, outdir, dpi, amp_cmap, plt) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comps = [("Ex", beam.Ex)]
    if beam.ispol == 1 and beam.Ey is not None:
        comps.append(("Ey", beam.Ey))

    quantities = [
        ("amplitude", lambda E: np.abs(E), "|{c}|", amp_cmap, None, None),
        ("phase", lambda E: np.angle(E), "arg({c}) (rad)", "twilight",
         -np.pi, np.pi),
        ("intensity", lambda E: np.abs(E) ** 2, "|{c}|$^2$ (arb. u.)",
         amp_cmap, None, None),
    ]

    saved: list[Path] = []

    # individual figures
    for cname, E in comps:
        for qname, func, label_fmt, cmap, vmin, vmax in quantities:
            fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
            label = label_fmt.format(c=cname)
            _save_map((fig, ax), func(E), beam, f"{cname} {qname}", label,
                      cmap, vmin, vmax)
            path = outdir / f"{cname}_{qname}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    # combined overview (manual spacing: constrained layout does not account
    # for the divider colorbar axes and lets the panels collide)
    nrows = len(comps)
    fig, axes = plt.subplots(nrows, 3, figsize=(16.5, 4.8 * nrows),
                             squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.92,
                        wspace=0.65, hspace=0.40)
    for r, (cname, E) in enumerate(comps):
        for c, (qname, func, label_fmt, cmap, vmin, vmax) in enumerate(quantities):
            _save_map((fig, axes[r][c]), func(E), beam,
                      f"{cname} {qname}", label_fmt.format(c=cname),
                      cmap, vmin, vmax,
                      show_xlabel=(r == nrows - 1), show_ylabel=(c == 0))
    path = outdir / "summary.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read a ZEMAX beam file (ZBF) and plot amplitude / "
                    "phase / intensity of Ex (and Ey if polarized).")
    parser.add_argument("zbf_file", help="input .zbf file")
    parser.add_argument("-o", "--outdir", default=None,
                        help="output folder for the PNG maps "
                             "(default: <zbf name>_plots next to the file)")
    parser.add_argument("--info", action="store_true",
                        help="print header info only, no plots")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="crop to the central 1/ZOOM of the grid before "
                             "plotting (e.g. 16 for tightly focused spots)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fontsize", type=float, default=16,
                        help="base font size in pt (default 16, slide-ready)")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    beam = read_zbf(args.zbf_file)
    print(f"== {args.zbf_file} ==")
    print(beam.describe())

    if args.info:
        return

    outdir = args.outdir or Path(args.zbf_file).with_suffix("").name + "_plots"
    if args.zoom > 1:
        beam = crop_center(beam, args.zoom)
        print(f"\nzoom x{args.zoom:g}: plotting central "
              f"{beam.nx} x {beam.ny} samples "
              f"({beam.nx * beam.dx:g} x {beam.ny * beam.dy:g} "
              f"{beam.unit_name})")
    saved = plot_beam(beam, outdir, dpi=args.dpi, fontsize=args.fontsize)
    print(f"\nsaved {len(saved)} images to {Path(outdir).resolve()}:")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
