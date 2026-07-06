"""
Laser-to-Metalens Collimator with an IDEAL HYPERBOLIC PHASE (f = 26.6 um in air).

Same pipeline and figures as Laser2Metalens_Collimator_MetaAtom.py, but the
collimating phase is NOT the joint phase conjugation — it is the analytic
equal-path (hyperbolic) lens phase with focal length 26.6 um in air:

    phi(r) = -2*pi/lambda * ( sqrt(r^2 + f^2) - f ),   f = 26.6 um

realized with the asia_1310 meta-atom database (MetaAtomElement), applied to
both polarizations (Ex, Ey) of the Zemax POP source after 40 um of
propagation in the glue (n = 1.5).

Run from the repository root:
    python Example/Laser2Metalens_Collimator_IdealPhase.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')  # no blocking windows
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

# Allow running the example without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import optiprop

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# The Zemax POP source listings are NOT distributed with the repository
# (>100 MB each, over the GitHub file limit). Export them from Zemax and
# place them under source/ before running this example.
SOURCE_DIR = os.path.join(REPO_ROOT, 'source')
_required = [f'P3_10um_{p}_{k}.txt' for p in ('EX', 'EY') for k in ('I', 'Phase')]
_missing = [f for f in _required if not os.path.exists(os.path.join(SOURCE_DIR, f))]
if _missing:
    sys.exit(
        'Missing Zemax POP source files (not distributed with the repository):\n  '
        + '\n  '.join(os.path.join('source', f) for f in _missing)
        + '\nExport the POP irradiance/phase listings from Zemax and place them under source/.'
    )

# ----------------------------------------------------------------------------
# Design parameters
# ----------------------------------------------------------------------------
DESIGN_LAMBDA = 1.31e-6        # m (vacuum wavelength)
N_GLUE = 1.5                   # refractive index of the glue
GLUE_DISTANCE = 40e-6          # m, source -> metalens
LENS_DIAMETER = 90e-6          # m
FOCAL_LENGTH = 26.6e-6         # m, ideal hyperbolic lens focal length (in air)
PIXEL_SIZE = 325e-9            # m (= half of the 650 nm meta-atom period)
FIELD_L = 160e-6               # m simulation window
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

near_field = optiprop.NearField(
    pixel_size=PIXEL_SIZE,
    field_Lx=FIELD_L,
    field_Ly=FIELD_L,
    device=DEVICE,
)

# ----------------------------------------------------------------------------
# 1-2. Load the Zemax POP sources (Ex, Ey) and propagate 40 um in the glue
# ----------------------------------------------------------------------------
def load_and_propagate(pol):
    src = optiprop.ZemaxPOPSource(near_field)
    src.calculate_phase(
        intensity_file=os.path.join(REPO_ROOT, 'source', f'P3_10um_{pol}_I.txt'),
        phase_file=os.path.join(REPO_ROOT, 'source', f'P3_10um_{pol}_Phase.txt'),
    )
    prop = optiprop.ASMPropagation(
        propagation_wavelength=DESIGN_LAMBDA,
        propagation_distance=GLUE_DISTANCE,
        n=N_GLUE,
        device=DEVICE,
    )
    prop.set_input_field(u_in=src.U0, pixel_size=PIXEL_SIZE)
    prop.propagate()
    return src, prop.get_output_U


source_x, U_x = load_and_propagate('EX')
source_y, U_y = load_and_propagate('EY')
source_y.rich_print()
P_x = source_x.source_total_power
P_y = source_y.source_total_power
print(f'polarization power ratio Ey/Ex = {P_y / P_x:.1f}')

# Source amplitude and phase for both polarizations (at the source plane)
extent = [near_field.X.min().item()*1e6, near_field.X.max().item()*1e6,
          near_field.Y.min().item()*1e6, near_field.Y.max().item()*1e6]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for row, (pol, src) in enumerate([('Ex', source_x), ('Ey', source_y)]):
    amplitude = torch.abs(src.U0).cpu().numpy()
    phase = (torch.angle(src.U0) % (2*np.pi)).cpu().numpy()
    im0 = axes[row, 0].imshow(amplitude, extent=extent, cmap='turbo', origin='lower')
    axes[row, 0].set_title(f'{pol} amplitude (source plane)')
    fig.colorbar(im0, ax=axes[row, 0], label='Amplitude')
    im1 = axes[row, 1].imshow(phase, extent=extent, cmap='turbo', origin='lower',
                              vmin=0, vmax=2*np.pi)
    axes[row, 1].set_title(f'{pol} phase (source plane)')
    fig.colorbar(im1, ax=axes[row, 1], label='Phase (rad)')
    for ax in axes[row]:
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'collimator_ideal_source_ExEy.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. Ideal hyperbolic lens phase (f = 26.6 um in air) realized with meta-atoms
# ----------------------------------------------------------------------------
aperture = (near_field.X**2 + near_field.Y**2) <= (LENS_DIAMETER / 2)**2

ideal_lens = optiprop.EqualPathPhase(near_field)
ideal_lens.calculate_phase(
    focal_length=FOCAL_LENGTH,
    design_lambda=DESIGN_LAMBDA,
    lens_diameter=LENS_DIAMETER,
    aperture_type='circle',
    aperture_size=[LENS_DIAMETER],
)
ideal_lens.rich_print()

library = optiprop.MetaAtomLibrary(os.path.join(REPO_ROOT, 'asia_1310.npy'), device=DEVICE)
library.rich_print()

meta_lens = optiprop.MetaAtomElement(near_field)
meta_lens.calculate_phase(
    ideal_element=ideal_lens,
    library=library,
    alpha=0.3,
    optimize_global_offset=True,
)
meta_lens.rich_print()
meta_lens.export_parameter_map(os.path.join(OUTPUT_DIR, 'collimator_ideal_R_map.csv'))
meta_lens.draw_parameter_map(show=False, save_path=os.path.join(OUTPUT_DIR, 'collimator_ideal_R_map.png'))


def wavefront_rms(U):
    """Intensity-weighted RMS of the wrapped residual phase inside the aperture."""
    w = (torch.abs(U)**2)[aperture]
    ph = torch.angle(U)[aperture]
    mean_ph = torch.atan2((w*torch.sin(ph)).sum(), (w*torch.cos(ph)).sum())
    dph = torch.remainder(ph - mean_ph + torch.pi, 2*torch.pi) - torch.pi
    return torch.sqrt((w*dph**2).sum()/w.sum()).item()


# Wavefront flatness after the lens for both polarizations
for pol, U in [('Ex', U_x), ('Ey', U_y)]:
    for lens_name, L in [('ideal f=26.6', ideal_lens.U0), ('metaatom', meta_lens.U0), ('no lens', aperture)]:
        rms = wavefront_rms(U * L)
        print(f'wavefront RMS {pol} ({lens_name:12s}): {rms:.4f} rad = {rms/(2*np.pi):.4f} waves')


def beam_sigma(U, x):
    """Intensity-weighted second-moment widths (sigma_x, sigma_y) in m."""
    I = (torch.abs(U)**2).cpu().numpy()
    Ix, Iy = I.sum(axis=0), I.sum(axis=1)
    cx, cy = (Ix*x).sum()/Ix.sum(), (Iy*x).sum()/Iy.sum()
    sx = np.sqrt((Ix*(x-cx)**2).sum()/Ix.sum())
    sy = np.sqrt((Iy*(x-cy)**2).sum()/Iy.sum())
    return sx, sy


# ----------------------------------------------------------------------------
# 4. Collimation check: beam size vs z in air, with vs without the metalens
# ----------------------------------------------------------------------------
N_PAD = 985  # 985 * 325 nm = 320 um window
x_pad = (np.arange(N_PAD) - (N_PAD - 1)/2) * PIXEL_SIZE
z_list = np.linspace(0, 1000e-6, 11)
sigmas = {}
for pol, U in [('Ex', U_x), ('Ey', U_y)]:
    for lens_name, L in [('metaatom', meta_lens.U0), ('no lens', aperture)]:
        U_padded = optiprop.pad_to_center(U * L, N_PAD)
        sig = []
        for z in z_list:
            if z == 0:
                Uz = U_padded
            else:
                air_prop = optiprop.ASMPropagation(
                    propagation_wavelength=DESIGN_LAMBDA,
                    propagation_distance=float(z),
                    n=1.0,
                    device=DEVICE,
                )
                air_prop.set_input_field(u_in=U_padded, pixel_size=PIXEL_SIZE)
                air_prop.propagate()
                Uz = air_prop.get_output_U
            sig.append(beam_sigma(Uz, x_pad))
        sigmas[(pol, lens_name)] = np.array(sig)
        s = sigmas[(pol, lens_name)]
        print(f'{pol} {lens_name:9s} sigma_x: {s[0,0]*1e6:5.1f} -> {s[-1,0]*1e6:5.1f} um | '
              f'sigma_y: {s[0,1]*1e6:5.1f} -> {s[-1,1]*1e6:5.1f} um  (z: 0 -> 1000 um)')

# Beam size vs z figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
colors = {'Ex': 'tab:blue', 'Ey': 'tab:orange'}
for ax, comp, lbl in [(axes[0], 0, '$\\sigma_x$'), (axes[1], 1, '$\\sigma_y$')]:
    for pol in ['Ex', 'Ey']:
        ax.plot(z_list*1e6, sigmas[(pol, 'metaatom')][:, comp]*1e6, '-o',
                color=colors[pol], label=f'{pol} metalens')
        ax.plot(z_list*1e6, sigmas[(pol, 'no lens')][:, comp]*1e6, '--s',
                color=colors[pol], alpha=0.5, label=f'{pol} no lens')
    ax.set_xlabel('z in air after metalens (µm)')
    ax.set_title(lbl)
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel('beam size $\\sigma$ (µm)')
fig.suptitle('Ideal-phase collimator f = 26.6 µm (asia_1310): Ex and Ey polarizations')
fig.savefig(os.path.join(OUTPUT_DIR, 'collimator_ideal_sigma_vs_z.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. Save the output fields and plot the XZ intensity maps (Ex + Ey)
# ----------------------------------------------------------------------------
U_after_x = U_x * meta_lens.U0
U_after_y = U_y * meta_lens.U0
np.savez(
    os.path.join(OUTPUT_DIR, 'collimator_ideal_output_fields.npz'),
    U_after_Ex=U_after_x.cpu().numpy(),
    U_after_Ey=U_after_y.cpu().numpy(),
    pixel_size=PIXEL_SIZE,
    design_lambda=DESIGN_LAMBDA,
)
print('Saved fields after collimator to collimator_ideal_output_fields.npz '
      '(keys: U_after_Ex, U_after_Ey, pixel_size, design_lambda)')

# Propagate 0.2 um in air past the lens and save Ex/Ey as a .mat (complex double)
PROP_AFTER_LENS = 0.2e-6
mat_data = {'dx': PIXEL_SIZE, 'dy': PIXEL_SIZE,
            'Nx': int(near_field.Nx), 'Ny': int(near_field.Ny)}
for pol, U_after in [('EX', U_after_x), ('EY', U_after_y)]:
    p = optiprop.ASMPropagation(
        propagation_wavelength=DESIGN_LAMBDA,
        propagation_distance=PROP_AFTER_LENS,
        n=1.0,
        device=DEVICE,
    )
    p.set_input_field(u_in=U_after, pixel_size=PIXEL_SIZE)
    p.propagate()
    mat_data[pol] = p.get_output_U.cpu().numpy().astype(np.complex128)
sio.savemat(os.path.join(OUTPUT_DIR, 'collimator_ideal_output_0p2um.mat'), mat_data)
print('Saved fields 0.2 um after lens to collimator_ideal_output_0p2um.mat '
      '(keys: EX, EY [complex double], dx, dy, Nx, Ny)')

z_range = np.linspace(1e-6, 1000e-6, 201)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, (pol, U_after) in zip(axes, [('Ey', U_after_y), ('Ex', U_after_x)]):
    xz_prop = optiprop.ASMPropagation(
        propagation_wavelength=DESIGN_LAMBDA,
        propagation_distance=GLUE_DISTANCE,
        n=1.0,
        device=DEVICE,
    )
    xz_prop.set_input_field(u_in=optiprop.pad_to_center(U_after, N_PAD), pixel_size=PIXEL_SIZE)
    xz_prop.propagate_xz(z_range=z_range)
    I_xz = (torch.abs(xz_prop.output_UZ) ** 2).cpu().numpy()
    im = ax.imshow(
        I_xz.T,
        extent=[z_range[0]*1e6, z_range[-1]*1e6, x_pad[0]*1e6, x_pad[-1]*1e6],
        aspect='auto', origin='lower', cmap='turbo',
    )
    ax.set_ylabel('x (µm)')
    ax.set_ylim(-100, 100)
    ax.set_title(f'XZ intensity after metalens ({pol}, ideal phase f = 26.6 µm)')
    fig.colorbar(im, ax=ax, label='Intensity')
axes[-1].set_xlabel('z (µm)')
fig.savefig(os.path.join(OUTPUT_DIR, 'collimator_ideal_xz_ExEy.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Field snapshot at the lens plane (Ey)
optiprop.plot_field_intensity(
    U_y, near_field.X, near_field.Y,
    title='Ey at metalens plane (after 40 µm in glue)', show=False,
)
plt.savefig(os.path.join(OUTPUT_DIR, 'collimator_ideal_lens_plane.png'), dpi=300, bbox_inches='tight')
plt.close('all')

print('Figures and R map saved to:', OUTPUT_DIR)
