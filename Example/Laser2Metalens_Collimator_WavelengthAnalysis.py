"""
Wavelength-shift analysis of the metalens collimator designed at 1310 nm.

The collimator pillar arrangement (R per lattice site) is designed exactly as
in Laser2Metalens_Collimator_MetaAtom.py: joint (Ex,Ey) phase conjugation at
1310 nm realized with asia_1310.npy. This script then KEEPS that pillar
arrangement (the design index map) and asks how the same physical lens
performs at 1290 nm and 1330 nm: for every selected pillar the amplitude and
phase are re-looked-up in asia_1290.npy / asia_1330.npy, the source is
re-propagated through the glue at the shifted wavelength, and the collimation
metrics are re-evaluated.

Assumptions:
    - The source near-field profile (P3_10um POP export at 1310 nm) is reused
      unchanged at 1290/1330 nm; only the propagation wavelength and the
      meta-atom response change.
    - The glue index n = 1.5 is treated as non-dispersive.

Outputs (all written to <repo>/output/):
    - wl{1290,1310,1330}_before_lens.mat/.zbf : Ex, Ey at the metalens plane
      (after 65 um of glue, before the lens)
    - wl{1290,1310,1330}_after_lens_0p2um.mat/.zbf : Ex, Ey 0.2 um past the
      lens (in air)
      .mat keys: EX, EY (complex double), dx, dy (m), Nx, Ny
      .zbf: polarized (Ex+Ey), version 1, unit = mm (dx, dy, wavelength in mm)
    - wavelength_metalens_amp_phase.png : realized lens amplitude/phase at
      the three wavelengths (same pillars)
    - wavelength_sigma_vs_z.png : Ey (and Ex) beam size vs z in air
    - wavelength_xz_Ey.png : Ey XZ intensity maps
    - wavelength_summary.txt : metric table

Run from the repository root:
    python Example/Laser2Metalens_Collimator_WavelengthAnalysis.py
"""
import os
import sys
import argparse


def _parse_wavelengths(value):
    """Parse a comma-separated wavelength list expressed in nm."""
    try:
        values = [float(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('波長必須是以逗號分隔的數字（單位 nm）') from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError('請至少輸入一個大於 0 的波長')
    return values


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='Metalens collimator wavelength analysis and ZBF exporter.'
    )
    parser.add_argument('--design-wavelength-nm', type=float, default=1310.0)
    parser.add_argument('--wavelengths-nm', type=_parse_wavelengths,
                        default=_parse_wavelengths('1290,1310,1330'))
    parser.add_argument('--library-dir', default=None)
    parser.add_argument('--source-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--min-transmission', type=float, default=0.80)
    parser.add_argument('--glue-index', type=float, default=1.5)
    parser.add_argument('--glue-distance-um', type=float, default=125.0)
    parser.add_argument('--lens-diameter-um', type=float, default=90.0)
    parser.add_argument('--after-lens-um', type=float, default=0.2)
    parser.add_argument('--pixel-size-nm', type=float, default=325.0)
    parser.add_argument('--field-size-um', type=float, default=160.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--z-max-um', type=float, default=1000.0)
    parser.add_argument('--z-points', type=int, default=11)
    parser.add_argument('--xz-points', type=int, default=201)
    parser.add_argument('--pad-points', type=int, default=985)
    parser.add_argument('--no-mat', action='store_true')
    parser.add_argument('--no-zbf', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    return parser


ARGS = build_argument_parser().parse_args()

import matplotlib
matplotlib.use('Agg')  # no blocking windows
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.io as sio
import torch

# Slide-ready fonts for every figure in this script
FS = 16
plt.rcParams.update({
    'font.size': FS,
    'axes.titlesize': FS + 2,
    'axes.labelsize': FS + 1,
    'xtick.labelsize': FS - 1,
    'ytick.labelsize': FS - 1,
    'legend.fontsize': FS - 3,
})


def add_colorbar(fig, ax, im, label):
    """Colorbar on a divider axes: exactly as tall as the image axes."""
    cax = make_axes_locatable(ax).append_axes('right', size='4.5%', pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 2)
    return cb

# Allow running the example without installing the package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
import optiprop
from zbf import ZbfBeam, write_zbf

LIBRARY_DIR = os.path.abspath(ARGS.library_dir or os.path.join(REPO_ROOT, 'data', 'metaatoms'))
SOURCE_DIR = os.path.abspath(ARGS.source_dir or os.path.join(REPO_ROOT, 'source'))
OUTPUT_DIR = os.path.abspath(ARGS.output_dir or os.path.join(REPO_ROOT, 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Design parameters (identical to Laser2Metalens_Collimator_MetaAtom.py)
# ----------------------------------------------------------------------------
DESIGN_LAMBDA = ARGS.design_wavelength_nm * 1e-9
ANALYSIS_LAMBDAS = [value * 1e-9 for value in ARGS.wavelengths_nm]
if not any(np.isclose(DESIGN_LAMBDA, wavelength, rtol=0, atol=1e-15)
           for wavelength in ANALYSIS_LAMBDAS):
    raise ValueError('設計波長必須包含在分析波長清單中。')


def wavelength_filename_nm(wavelength_m):
    value = wavelength_m * 1e9
    return (
        str(int(round(value)))
        if np.isclose(value, round(value), rtol=0, atol=1e-9)
        else f'{value:g}'
    )


LIBRARIES = {
    wavelength: f'asia_{wavelength_filename_nm(wavelength)}.npy'
    for wavelength in ANALYSIS_LAMBDAS
}
# Reject a physical pillar if its intensity transmission falls below this
# value at ANY analysis wavelength.  It is replaced by the nearest-radius
# pillar that stays above the threshold across the full wavelength set.
MIN_PILLAR_TRANSMISSION = ARGS.min_transmission
N_GLUE = ARGS.glue_index
GLUE_DISTANCE = ARGS.glue_distance_um * 1e-6
LENS_DIAMETER = ARGS.lens_diameter_um * 1e-6
PROP_AFTER_LENS = ARGS.after_lens_um * 1e-6
PIXEL_SIZE = ARGS.pixel_size_nm * 1e-9
FIELD_L = ARGS.field_size_um * 1e-6
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') if ARGS.device == 'auto' else ARGS.device
if DEVICE == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('已指定 CUDA，但目前的 PyTorch 無法使用 CUDA。')
if not 0 <= MIN_PILLAR_TRANSMISSION <= 1:
    raise ValueError('最低穿透率必須介於 0 與 1 之間。')
if min(N_GLUE, GLUE_DISTANCE, LENS_DIAMETER, PIXEL_SIZE, FIELD_L) <= 0:
    raise ValueError('折射率、距離、口徑、像素與模擬範圍必須大於 0。')
if PROP_AFTER_LENS < 0:
    raise ValueError('Lens 後匯出位置不可小於 0。')
if ARGS.z_points < 2 or ARGS.xz_points < 2 or ARGS.pad_points < 2:
    raise ValueError('掃描點數至少為 2。')

# The Zemax POP source listings are NOT distributed with the repository.
_required = [f'P3_10um_{p}_{k}.txt' for p in ('EX', 'EY') for k in ('I', 'Phase')]
_missing = [f for f in _required if not os.path.exists(os.path.join(SOURCE_DIR, f))]
if _missing:
    sys.exit('Missing Zemax POP source files:\n  '
             + '\n  '.join(os.path.join('source', f) for f in _missing))

near_field = optiprop.NearField(
    pixel_size=PIXEL_SIZE,
    field_Lx=FIELD_L,
    field_Ly=FIELD_L,
    device=DEVICE,
)
aperture = (near_field.X**2 + near_field.Y**2) <= (LENS_DIAMETER / 2)**2

# ----------------------------------------------------------------------------
# Load the sources once (near-field profile assumed wavelength-independent)
# ----------------------------------------------------------------------------
def load_source(pol):
    src = optiprop.ZemaxPOPSource(near_field)
    src.calculate_phase(
        intensity_file=os.path.join(SOURCE_DIR, f'P3_10um_{pol}_I.txt'),
        phase_file=os.path.join(SOURCE_DIR, f'P3_10um_{pol}_Phase.txt'),
    )
    return src


source_x = load_source('EX')
source_y = load_source('EY')


def propagate_glue(U0, wavelength):
    prop = optiprop.ASMPropagation(
        propagation_wavelength=wavelength,
        propagation_distance=GLUE_DISTANCE,
        n=N_GLUE,
        device=DEVICE,
    )
    prop.set_input_field(u_in=U0, pixel_size=PIXEL_SIZE)
    prop.propagate()
    return prop.get_output_U


# ----------------------------------------------------------------------------
# 1. Design at 1310 nm: joint phase conjugation + asia_1310 lookup
#    (identical to Laser2Metalens_Collimator_MetaAtom.py)
# ----------------------------------------------------------------------------
U_x_1310 = propagate_glue(source_x.U0, DESIGN_LAMBDA)
U_y_1310 = propagate_glue(source_y.U0, DESIGN_LAMBDA)
P_x = source_x.source_total_power
P_y = source_y.source_total_power

I_x, I_y = torch.abs(U_x_1310)**2, torch.abs(U_y_1310)**2
psi_x, psi_y = torch.angle(U_x_1310), torch.angle(U_y_1310)


def wavefront_rms(U):
    """Intensity-weighted RMS of the wrapped residual phase inside the aperture."""
    w = (torch.abs(U)**2)[aperture]
    ph = torch.angle(U)[aperture]
    mean_ph = torch.atan2((w*torch.sin(ph)).sum(), (w*torch.cos(ph)).sum())
    dph = torch.remainder(ph - mean_ph + torch.pi, 2*torch.pi) - torch.pi
    return torch.sqrt((w*dph**2).sum()/w.sum()).item()


best_delta, best_cost = 0.0, float('inf')
for delta in np.linspace(0, 2*np.pi, 181):
    phasor = I_x * torch.exp(1j*psi_x) + I_y * torch.exp(1j*(psi_y + delta))
    lens_U0 = aperture * torch.exp(-1j * torch.angle(phasor))
    cost = P_x * wavefront_rms(U_x_1310*lens_U0)**2 + P_y * wavefront_rms(U_y_1310*lens_U0)**2
    if cost < best_cost:
        best_delta, best_cost = float(delta), cost

phasor = I_x * torch.exp(1j*psi_x) + I_y * torch.exp(1j*(psi_y + best_delta))
ideal_U0 = aperture * torch.exp(-1j * torch.angle(phasor))

design_library_key = min(ANALYSIS_LAMBDAS, key=lambda wl: abs(wl - DESIGN_LAMBDA))
library_1310 = optiprop.MetaAtomLibrary(
    os.path.join(LIBRARY_DIR, LIBRARIES[design_library_key]),
    device=DEVICE,
)
meta_lens = optiprop.MetaAtomElement(near_field)
meta_lens.calculate_phase(
    ideal_U0=ideal_U0,
    library=library_1310,
    alpha=ARGS.alpha,
    optimize_global_offset=True,
)
index_map = meta_lens.index_map          # pillar choice per pixel (the fixed layout)
print(f'design @{ARGS.design_wavelength_nm:g}nm: delta={best_delta:.4f} rad, '
      f'global offset={meta_lens.global_offset:.4f} rad')

# Avoid narrow resonant pillars that look efficient at 1310 nm but collapse at
# an adjacent wavelength.  Among pillars meeting the transmission floor at all
# wavelengths, use the closest radius; if two radii are equally close, choose
# the one with the smaller 1310-nm phase change.
libraries = {}
for wl in ANALYSIS_LAMBDAS:
    lib = optiprop.MetaAtomLibrary(
        os.path.join(LIBRARY_DIR, LIBRARIES[wl]),
        device=DEVICE,
    )
    if not torch.equal(lib.parameter, library_1310.parameter):
        raise ValueError(f'R axis of {LIBRARIES[wl]} differs from the design library.')
    libraries[wl] = lib

min_transmission_by_radius = torch.stack(
    [libraries[wl].amplitude**2 for wl in ANALYSIS_LAMBDAS]
).amin(dim=0)
safe_indices = torch.where(min_transmission_by_radius >= MIN_PILLAR_TRANSMISSION)[0]
if safe_indices.numel() == 0:
    raise ValueError(
        f'No pillar has intensity transmission >= {MIN_PILLAR_TRANSMISSION:.2f} '
        'at every analysis wavelength.'
    )

radii = library_1310.parameter
radius_distance = torch.abs(radii[:, None] - radii[safe_indices][None, :])
nearest_distance = radius_distance.amin(dim=1, keepdim=True)
same_distance = torch.isclose(radius_distance, nearest_distance)
phase_change = torch.abs(torch.remainder(
    library_1310.phase[safe_indices][None, :] - library_1310.phase[:, None]
    + torch.pi, 2*torch.pi
) - torch.pi)
tie_break_cost = torch.where(
    same_distance, phase_change, torch.full_like(phase_change, torch.inf)
)
replacement_by_index = safe_indices[tie_break_cost.argmin(dim=1)]

original_index_map = index_map.clone()
low_transmission_pixels = aperture & (
    min_transmission_by_radius[index_map] < MIN_PILLAR_TRANSMISSION
)
index_map = torch.where(
    low_transmission_pixels, replacement_by_index[index_map], index_map
)
n_replaced = int(low_transmission_pixels.sum().item())
center_flat = int(torch.argmin(near_field.X**2 + near_field.Y**2).item())
center_iy, center_ix = np.unravel_index(center_flat, index_map.shape)
old_center_idx = int(original_index_map[center_iy, center_ix].item())
new_center_idx = int(index_map[center_iy, center_ix].item())
print(
    f'broadband transmission guard: T >= {MIN_PILLAR_TRANSMISSION:.2f}, '
    f'replaced {n_replaced} pixels; center R '
    f'{radii[old_center_idx].item():.0f} -> {radii[new_center_idx].item():.0f} nm'
)

# ----------------------------------------------------------------------------
# 2. Re-evaluate the SAME pillar layout at 1290 / 1310 / 1330 nm
# ----------------------------------------------------------------------------
N_PAD = ARGS.pad_points
if N_PAD < max(int(near_field.Nx), int(near_field.Ny)):
    raise ValueError(
        f'pad-points ({N_PAD}) 不可小於計算網格 '
        f'({max(int(near_field.Nx), int(near_field.Ny))})。'
    )
x_pad = (np.arange(N_PAD) - (N_PAD - 1)/2) * PIXEL_SIZE
z_list = np.linspace(0, ARGS.z_max_um * 1e-6, ARGS.z_points)


def beam_sigma(U, x):
    I = (torch.abs(U)**2).cpu().numpy()
    Ix_, Iy_ = I.sum(axis=0), I.sum(axis=1)
    cx, cy = (Ix_*x).sum()/Ix_.sum(), (Iy_*x).sum()/Iy_.sum()
    return (np.sqrt((Ix_*(x-cx)**2).sum()/Ix_.sum()),
            np.sqrt((Iy_*(x-cy)**2).sum()/Iy_.sum()))


def save_mat(path, Ex, Ey):
    sio.savemat(path, {
        'EX': Ex.cpu().numpy().astype(np.complex128),
        'EY': Ey.cpu().numpy().astype(np.complex128),
        'dx': PIXEL_SIZE, 'dy': PIXEL_SIZE,
        'Nx': int(near_field.Nx), 'Ny': int(near_field.Ny),
    })


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def save_zbf(path, Ex, Ey, wavelength_m, medium_index):
    """Write a polarized version-1 ZBF. ALL lengths in mm (Zemax lens units).

    ZBF grids must be a power of two (256/512/1024/...), so the field is
    zero-padded to the next power of two with the same pixel size.
    """
    N_ZBF = _next_pow2(max(int(near_field.Nx), int(near_field.Ny)))
    Ex_np = optiprop.pad_to_center(Ex, N_ZBF).cpu().numpy().astype(np.complex128)
    Ey_np = optiprop.pad_to_center(Ey, N_ZBF).cpu().numpy().astype(np.complex128)
    # Pilot beam from the dominant (Ey) intensity: waist ~ sigma at this plane
    sx, sy = beam_sigma(Ey, (np.arange(near_field.Nx) - (near_field.Nx - 1)/2) * PIXEL_SIZE)
    mm = 1e3  # m -> mm
    from zbf import PilotRays
    pilot = PilotRays(
        posX=0.0, posY=0.0,
        waistX=sx * mm, waistY=sy * mm,
        rayleighX=np.pi * sx**2 * medium_index / wavelength_m * mm,
        rayleighY=np.pi * sy**2 * medium_index / wavelength_m * mm,
    )
    beam = ZbfBeam(
        nx=N_ZBF, ny=N_ZBF,
        dx=PIXEL_SIZE * mm, dy=PIXEL_SIZE * mm,   # mm
        ispol=1,
        lam=wavelength_m * mm,                    # vacuum wavelength in mm
        Ex=Ex_np, Ey=Ey_np,
        pilot=pilot,
        unit=0,                                   # 0 = mm
        index=medium_index,
    )
    write_zbf(beam, path)


results = {}
lens_U0_by_wl = {}
for wl in ANALYSIS_LAMBDAS:
    wl_nm = wavelength_filename_nm(wl)
    library = libraries[wl]

    # Same pillars (index_map), response of THIS wavelength's database
    amp_wl = library.amplitude[index_map] * aperture
    phase_wl = library.phase[index_map] * aperture
    lens_U0 = amp_wl * torch.exp(1j * phase_wl) * aperture
    lens_U0_by_wl[wl_nm] = lens_U0

    # Source propagated through the glue at this wavelength
    U_x = propagate_glue(source_x.U0, wl)
    U_y = propagate_glue(source_y.U0, wl)
    U_after_x = U_x * lens_U0
    U_after_y = U_y * lens_U0

    # Collimation metrics
    rms_x = wavefront_rms(U_after_x)
    rms_y = wavefront_rms(U_after_y)
    amp_in = amp_wl[aperture]
    # Transmitted power fraction of Ey through the lens (relative to incident in aperture)
    p_in = (torch.abs(U_y)**2)[aperture].sum().item()
    p_out = (torch.abs(U_after_y)**2)[aperture].sum().item()

    sig = {}
    for pol, U_after in [('Ex', U_after_x), ('Ey', U_after_y)]:
        U_padded = optiprop.pad_to_center(U_after, N_PAD)
        rows = []
        for z in z_list:
            if z == 0:
                Uz = U_padded
            else:
                p = optiprop.ASMPropagation(propagation_wavelength=wl,
                                            propagation_distance=float(z),
                                            n=1.0, device=DEVICE)
                p.set_input_field(u_in=U_padded, pixel_size=PIXEL_SIZE)
                p.propagate()
                Uz = p.get_output_U
            rows.append(beam_sigma(Uz, x_pad))
        sig[pol] = np.array(rows)

    # XZ map (Ey)
    I_xz = None
    if not ARGS.no_plots:
        xz = optiprop.ASMPropagation(propagation_wavelength=wl,
                                     propagation_distance=GLUE_DISTANCE,
                                     n=1.0, device=DEVICE)
        xz.set_input_field(u_in=optiprop.pad_to_center(U_after_y, N_PAD),
                           pixel_size=PIXEL_SIZE)
        xz.propagate_xz(z_range=np.linspace(
            min(1e-6, ARGS.z_max_um * 1e-6),
            ARGS.z_max_um * 1e-6,
            ARGS.xz_points,
        ))
        I_xz = (torch.abs(xz.output_UZ)**2).cpu().numpy()

    results[wl_nm] = {
        'rms_x': rms_x, 'rms_y': rms_y,
        'amp_min': amp_in.min().item(), 'amp_mean': amp_in.mean().item(),
        'transmission_Ey': p_out / p_in,
        'sig': sig, 'I_xz': I_xz,
    }
    print(f'{wl_nm}nm: Ey RMS {rms_y:.4f} rad ({rms_y/2/np.pi:.4f} waves) | '
          f'lens amp min/mean {amp_in.min().item():.3f}/{amp_in.mean().item():.3f} | '
          f'Ey power transmission {p_out/p_in:.3f} | '
          f'Ey sigma_x @{ARGS.z_max_um:g}um {sig["Ey"][-1,0]*1e6:.1f} um')

    # .mat + .zbf exports (before lens: in glue; after lens: 0.2 um into air)
    if not ARGS.no_mat:
        save_mat(os.path.join(OUTPUT_DIR, f'wl{wl_nm}_before_lens.mat'), U_x, U_y)
    if not ARGS.no_zbf:
        save_zbf(os.path.join(OUTPUT_DIR, f'wl{wl_nm}_before_lens.zbf'),
                 U_x, U_y, wl, N_GLUE)

    after = {}
    for pol, U_after in [('EX', U_after_x), ('EY', U_after_y)]:
        p = optiprop.ASMPropagation(propagation_wavelength=wl,
                                    propagation_distance=PROP_AFTER_LENS,
                                    n=1.0, device=DEVICE)
        p.set_input_field(u_in=U_after, pixel_size=PIXEL_SIZE)
        p.propagate()
        after[pol] = p.get_output_U
    distance_tag = f'{ARGS.after_lens_um:g}'.replace('.', 'p')
    if not ARGS.no_mat:
        save_mat(os.path.join(OUTPUT_DIR, f'wl{wl_nm}_after_lens_{distance_tag}um.mat'),
                 after['EX'], after['EY'])
    if not ARGS.no_zbf:
        save_zbf(os.path.join(OUTPUT_DIR, f'wl{wl_nm}_after_lens_{distance_tag}um.zbf'),
                 after['EX'], after['EY'], wl, 1.0)

# ----------------------------------------------------------------------------
# 3. Comparison figures
# ----------------------------------------------------------------------------
extent = [near_field.X.min().item()*1e6, near_field.X.max().item()*1e6,
          near_field.Y.min().item()*1e6, near_field.Y.max().item()*1e6]

# Realized lens amplitude/phase at all analysis wavelengths (same pillars)
figure_columns = len(lens_U0_by_wl)
fig, axes = plt.subplots(
    2, figure_columns,
    figsize=(max(7, 6.3 * figure_columns), 11),
    squeeze=False,
)
fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.90,
                    wspace=0.45, hspace=0.30)
for col, wl_nm in enumerate(sorted(lens_U0_by_wl, key=float)):
    U0 = lens_U0_by_wl[wl_nm]
    amp = torch.abs(U0).cpu().numpy()
    ph = (torch.angle(U0) % (2*np.pi)).cpu().numpy()
    im0 = axes[0, col].imshow(amp, extent=extent, cmap='turbo', origin='lower',
                              vmin=0, vmax=1)
    axes[0, col].set_title(f'{wl_nm} nm lens amplitude')
    add_colorbar(fig, axes[0, col], im0, 'Amplitude')
    im1 = axes[1, col].imshow(ph, extent=extent, cmap='turbo', origin='lower',
                              vmin=0, vmax=2*np.pi)
    axes[1, col].set_title(f'{wl_nm} nm lens phase')
    add_colorbar(fig, axes[1, col], im1, 'Phase (rad)')
    for ax in axes[:, col]:
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
fig.suptitle(
    f'Same pillar layout (designed @{ARGS.design_wavelength_nm:g} nm): '
    'realized lens response vs wavelength'
)
if not ARGS.no_plots:
    fig.savefig(os.path.join(OUTPUT_DIR, 'wavelength_metalens_amp_phase.png'),
                dpi=300, bbox_inches='tight')
plt.close(fig)

# Beam size vs z
color_values = plt.cm.viridis(np.linspace(0.12, 0.88, max(1, len(results))))
colors = {
    wl_nm: color_values[index]
    for index, wl_nm in enumerate(sorted(results, key=float))
}
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, comp, lbl in [(axes[0], 0, '$\\sigma_x$'), (axes[1], 1, '$\\sigma_y$')]:
    for wl_nm in sorted(results, key=float):
        ax.plot(z_list*1e6, results[wl_nm]['sig']['Ey'][:, comp]*1e6, '-o',
                color=colors[wl_nm], label=f'Ey {wl_nm} nm')
        ax.plot(z_list*1e6, results[wl_nm]['sig']['Ex'][:, comp]*1e6, '--',
                color=colors[wl_nm], alpha=0.4, label=f'Ex {wl_nm} nm')
    ax.set_xlabel('z in air after metalens (µm)')
    ax.set_title(lbl)
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel('beam size $\\sigma$ (µm)')
fig.suptitle(
    f'Collimation vs wavelength (pillar layout designed '
    f'@{ARGS.design_wavelength_nm:g} nm)'
)
fig.tight_layout()
if not ARGS.no_plots:
    fig.savefig(os.path.join(OUTPUT_DIR, 'wavelength_sigma_vs_z.png'),
                dpi=300, bbox_inches='tight')
plt.close(fig)

# XZ maps (Ey)
figure_rows = len(results)
fig, axes = plt.subplots(
    figure_rows, 1,
    figsize=(15, max(5, 4 * figure_rows)),
    sharex=True,
    squeeze=False,
)
axes = axes[:, 0]
fig.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.95, hspace=0.35)
for ax, wl_nm in zip(axes, sorted(results, key=float)):
    if ARGS.no_plots:
        break
    im = ax.imshow(results[wl_nm]['I_xz'].T,
                   extent=[min(1.0, ARGS.z_max_um), ARGS.z_max_um,
                           x_pad[0]*1e6, x_pad[-1]*1e6],
                   aspect='auto', origin='lower', cmap='turbo')
    ax.set_ylabel('x (µm)')
    ax.set_ylim(-100, 100)
    ax.set_title(
        f'Ey XZ intensity, {wl_nm} nm '
        f'(layout designed @{ARGS.design_wavelength_nm:g} nm)'
    )
    add_colorbar(fig, ax, im, 'Intensity')
axes[-1].set_xlabel('z (µm)')
if not ARGS.no_plots:
    fig.savefig(os.path.join(OUTPUT_DIR, 'wavelength_xz_Ey.png'),
                dpi=300, bbox_inches='tight')
plt.close(fig)

# Summary table
lines = [f'Wavelength analysis: pillar layout designed @{ARGS.design_wavelength_nm:g} nm '
         f'(joint conjugation, alpha={ARGS.alpha:g})',
         f'lens diameter {LENS_DIAMETER*1e6:.0f} um, glue n={N_GLUE}, source->lens {GLUE_DISTANCE*1e6:.0f} um',
         f'broadband pillar guard: intensity T >= {MIN_PILLAR_TRANSMISSION:.2f} at '
         f'{"/".join(wavelength_filename_nm(wl) for wl in ANALYSIS_LAMBDAS)} nm; '
         f'replaced {n_replaced} pixels; center R {radii[old_center_idx].item():.0f} -> '
         f'{radii[new_center_idx].item():.0f} nm',
         '',
         f'wl(nm)  EyRMS(rad)  EyRMS(waves)  ExRMS(rad)  lensAmpMin  lensAmpMean  '
         f'EyPowerTrans  Ey_sx@{ARGS.z_max_um:g}um(um)  Ey_sy@{ARGS.z_max_um:g}um(um)']
for wl_nm in sorted(results, key=float):
    r = results[wl_nm]
    lines.append(f'{wl_nm:>6}  {r["rms_y"]:.4f}      {r["rms_y"]/2/np.pi:.4f}        '
                 f'{r["rms_x"]:.4f}      {r["amp_min"]:.3f}       {r["amp_mean"]:.3f}        '
                 f'{r["transmission_Ey"]:.3f}         {r["sig"]["Ey"][-1,0]*1e6:6.1f}         '
                 f'{r["sig"]["Ey"][-1,1]*1e6:6.1f}')
summary = '\n'.join(lines)
with open(os.path.join(OUTPUT_DIR, 'wavelength_summary.txt'), 'w') as f:
    f.write(summary + '\n')
print()
print(summary)
print('\nAll outputs saved to:', OUTPUT_DIR)
