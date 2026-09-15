"""
Wavelength-shift analysis of the IDEAL-PHASE metalens collimator (f = 36 um).

Counterpart of Laser2Metalens_Collimator_WavelengthAnalysis.py: the pillar
arrangement is designed exactly as in Laser2Metalens_Collimator_IdealPhase.py
— the analytic equal-path (hyperbolic) lens phase with focal length 36 um in
air (the scanned optimum for Ey), realized with asia_1310.npy at 1310 nm.
The pillar layout (design index map) is then KEPT FIXED and re-evaluated at
1290 nm and 1330 nm: for every selected pillar the amplitude and phase are
re-looked-up in asia_1290.npy / asia_1330.npy, the source is re-propagated
through the glue at the shifted wavelength, and the collimation metrics are
re-evaluated.

Assumptions:
    - The source near-field profile (P3_10um POP export at 1310 nm) is reused
      unchanged at 1290/1330 nm; only the propagation wavelength and the
      meta-atom response change.
    - The glue index n = 1.5 is treated as non-dispersive.

Outputs (all written to <repo>/output2/):
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
    python Example/Laser2Metalens_Collimator_WavelengthAnalysis_IdealPhase.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')  # no blocking windows
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.io as sio
import torch

# Allow running the example without installing the package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
import optiprop
from zbf import ZbfBeam, PilotRays, write_zbf

OUTPUT_DIR = os.path.join(REPO_ROOT, 'output2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# ----------------------------------------------------------------------------
# Design parameters (identical to Laser2Metalens_Collimator_IdealPhase.py)
# ----------------------------------------------------------------------------
DESIGN_LAMBDA = 1.31e-6        # m, design (vacuum) wavelength
ANALYSIS_LAMBDAS = [1.29e-6, 1.31e-6, 1.33e-6]   # m
LIBRARIES = {1.29e-6: 'asia_1290.npy',
             1.31e-6: 'asia_1310.npy',
             1.33e-6: 'asia_1330.npy'}
N_GLUE = 1.5                   # refractive index of the glue
GLUE_DISTANCE = 65e-6          # m, source -> metalens
LENS_DIAMETER = 90e-6          # m
FOCAL_LENGTH = 36e-6           # m, best hyperbolic focal length for Ey at 65 um
                               # glue (scanned: RMS 0.065 waves; f=16um gives
                               # 0.241, f=21um 0.215, geometric 65/1.5=43.3um 0.102)
PROP_AFTER_LENS = 0.2e-6       # m, air propagation for the "after lens" export
PIXEL_SIZE = 325e-9            # m (= half of the 650 nm meta-atom period)
FIELD_L = 160e-6               # m simulation window
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# The Zemax POP source listings are NOT distributed with the repository.
SOURCE_DIR = os.path.join(REPO_ROOT, 'source')
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


def wavefront_rms(U):
    """Intensity-weighted RMS of the wrapped residual phase inside the aperture."""
    w = (torch.abs(U)**2)[aperture]
    ph = torch.angle(U)[aperture]
    mean_ph = torch.atan2((w*torch.sin(ph)).sum(), (w*torch.cos(ph)).sum())
    dph = torch.remainder(ph - mean_ph + torch.pi, 2*torch.pi) - torch.pi
    return torch.sqrt((w*dph**2).sum()/w.sum()).item()


# ----------------------------------------------------------------------------
# 1. Design at 1310 nm: ideal hyperbolic phase (f = 36 um) + asia_1310 lookup
#    (identical to Laser2Metalens_Collimator_IdealPhase.py)
# ----------------------------------------------------------------------------
ideal_lens = optiprop.EqualPathPhase(near_field)
ideal_lens.calculate_phase(
    focal_length=FOCAL_LENGTH,
    design_lambda=DESIGN_LAMBDA,
    lens_diameter=LENS_DIAMETER,
    aperture_type='circle',
    aperture_size=[LENS_DIAMETER],
)

library_1310 = optiprop.MetaAtomLibrary(os.path.join(REPO_ROOT, 'data', 'metaatoms', LIBRARIES[DESIGN_LAMBDA]),
                                        device=DEVICE)
meta_lens = optiprop.MetaAtomElement(near_field)
meta_lens.calculate_phase(
    ideal_element=ideal_lens,
    library=library_1310,
    alpha=0.3,
    optimize_global_offset=True,
)
index_map = meta_lens.index_map          # pillar choice per pixel (the fixed layout)
print(f'design @1310nm: ideal hyperbolic f={FOCAL_LENGTH*1e6:.1f} um, '
      f'global offset={meta_lens.global_offset:.4f} rad')

# ----------------------------------------------------------------------------
# 2. Re-evaluate the SAME pillar layout at 1290 / 1310 / 1330 nm
# ----------------------------------------------------------------------------
N_PAD = 985  # 985 * 325 nm = 320 um window for the z-scan
x_pad = (np.arange(N_PAD) - (N_PAD - 1)/2) * PIXEL_SIZE
z_list = np.linspace(0, 1000e-6, 11)


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
    wl_nm = int(round(wl * 1e9))
    library = optiprop.MetaAtomLibrary(os.path.join(REPO_ROOT, 'data', 'metaatoms', LIBRARIES[wl]), device=DEVICE)
    if not torch.equal(library.parameter, library_1310.parameter):
        raise ValueError(f'R axis of {LIBRARIES[wl]} differs from the design library.')

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
    xz = optiprop.ASMPropagation(propagation_wavelength=wl,
                                 propagation_distance=GLUE_DISTANCE,
                                 n=1.0, device=DEVICE)
    xz.set_input_field(u_in=optiprop.pad_to_center(U_after_y, N_PAD), pixel_size=PIXEL_SIZE)
    xz.propagate_xz(z_range=np.linspace(1e-6, 1000e-6, 201))
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
          f'Ey sigma_x @1mm {sig["Ey"][-1,0]*1e6:.1f} um')

    # .mat + .zbf exports (before lens: in glue; after lens: 0.2 um into air)
    save_mat(os.path.join(OUTPUT_DIR, f'wl{wl_nm}_before_lens.mat'), U_x, U_y)
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
    save_mat(os.path.join(OUTPUT_DIR, f'wl{wl_nm}_after_lens_0p2um.mat'),
             after['EX'], after['EY'])
    save_zbf(os.path.join(OUTPUT_DIR, f'wl{wl_nm}_after_lens_0p2um.zbf'),
             after['EX'], after['EY'], wl, 1.0)

# ----------------------------------------------------------------------------
# 3. Comparison figures
# ----------------------------------------------------------------------------
extent = [near_field.X.min().item()*1e6, near_field.X.max().item()*1e6,
          near_field.Y.min().item()*1e6, near_field.Y.max().item()*1e6]

# Realized lens amplitude/phase at the three wavelengths (same pillars)
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.90,
                    wspace=0.45, hspace=0.30)
for col, wl_nm in enumerate(sorted(lens_U0_by_wl)):
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
fig.suptitle(f'Ideal-phase layout f={FOCAL_LENGTH*1e6:.0f} µm (designed @1310 nm): '
             'realized lens response vs wavelength')
fig.savefig(os.path.join(OUTPUT_DIR, 'wavelength_metalens_amp_phase.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)

# Beam size vs z
colors = {1290: 'tab:blue', 1310: 'tab:green', 1330: 'tab:red'}
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, comp, lbl in [(axes[0], 0, '$\\sigma_x$'), (axes[1], 1, '$\\sigma_y$')]:
    for wl_nm in sorted(results):
        ax.plot(z_list*1e6, results[wl_nm]['sig']['Ey'][:, comp]*1e6, '-o',
                color=colors[wl_nm], label=f'Ey {wl_nm} nm')
        ax.plot(z_list*1e6, results[wl_nm]['sig']['Ex'][:, comp]*1e6, '--',
                color=colors[wl_nm], alpha=0.4, label=f'Ex {wl_nm} nm')
    ax.set_xlabel('z in air after metalens (µm)')
    ax.set_title(lbl)
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel('beam size $\\sigma$ (µm)')
fig.suptitle(f'Collimation vs wavelength (ideal-phase layout f={FOCAL_LENGTH*1e6:.0f} µm @1310 nm)')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'wavelength_sigma_vs_z.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)

# XZ maps (Ey)
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.95, hspace=0.35)
for ax, wl_nm in zip(axes, sorted(results)):
    im = ax.imshow(results[wl_nm]['I_xz'].T,
                   extent=[1, 1000, x_pad[0]*1e6, x_pad[-1]*1e6],
                   aspect='auto', origin='lower', cmap='turbo')
    ax.set_ylabel('x (µm)')
    ax.set_ylim(-100, 100)
    ax.set_title(f'Ey XZ intensity, {wl_nm} nm (ideal-phase layout @1310 nm)')
    add_colorbar(fig, ax, im, 'Intensity')
axes[-1].set_xlabel('z (µm)')
fig.savefig(os.path.join(OUTPUT_DIR, 'wavelength_xz_Ey.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)

# Summary table
lines = [f'Wavelength analysis: IDEAL-PHASE pillar layout f={FOCAL_LENGTH*1e6:.0f} um '
         '(hyperbolic, designed @1310 nm, alpha=0.3)',
         f'lens diameter {LENS_DIAMETER*1e6:.0f} um, glue n={N_GLUE}, source->lens {GLUE_DISTANCE*1e6:.0f} um',
         '',
         'wl(nm)  EyRMS(rad)  EyRMS(waves)  ExRMS(rad)  lensAmpMin  lensAmpMean  EyPowerTrans  Ey_sx@1mm(um)  Ey_sy@1mm(um)']
for wl_nm in sorted(results):
    r = results[wl_nm]
    lines.append(f'{wl_nm:6d}  {r["rms_y"]:.4f}      {r["rms_y"]/2/np.pi:.4f}        '
                 f'{r["rms_x"]:.4f}      {r["amp_min"]:.3f}       {r["amp_mean"]:.3f}        '
                 f'{r["transmission_Ey"]:.3f}         {r["sig"]["Ey"][-1,0]*1e6:6.1f}         '
                 f'{r["sig"]["Ey"][-1,1]*1e6:6.1f}')
summary = '\n'.join(lines)
with open(os.path.join(OUTPUT_DIR, 'wavelength_summary.txt'), 'w') as f:
    f.write(summary + '\n')
print()
print(summary)
print('\nAll outputs saved to:', OUTPUT_DIR)
