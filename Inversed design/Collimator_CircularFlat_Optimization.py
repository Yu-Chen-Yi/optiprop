"""
Auto-gradient (torch autograd) beam-shaping optimization of the collimator phase.

Goal: the P3_10um source propagates 40 um through the glue (n = 1.5) and hits
a single phase distribution (90 um aperture). After the phase plate the beam
(dominant Ey polarization) should be CIRCULAR in amplitude and FLAT in phase.

A phase-only element cannot change the amplitude in its own plane, so the
objective is evaluated after propagation: at several planes z_k in air the
optimizer matches the beam to a collimated circular Gaussian mode
(waist w0 at the lens plane):

    L = sum_k [ || |U_k|/|||U_k||| - T_k/||T_k|| ||^2      (circular amplitude)
                + gamma * (1 - |sum U_k|^2 / (sum |U_k|)^2) ]   (flat phase)

where T_k is the analytic amplitude of the collimated Gaussian at z_k and the
flatness term is 1 - (coherent sum / incoherent sum)^2, which is 1 exactly
when the phase is uniform. Both terms are differentiable (no angle()).

The phase is initialized with the Ey phase conjugation (the previous best
design) and optimized with Adam through a differentiable ASM propagator.
Afterwards the optimized phase is realized with the asia_1310 meta-atom
database and re-evaluated.

Outputs (all written to <repo>/output3/):
    - optimized_phase.npy / .mat  : optimized phase map (rad) + grid info
    - optimization_loss.png       : loss / metric history
    - amp_vs_target.png           : |Ey| at the eval planes vs the target
    - sigma_vs_z.png              : sigma_x / sigma_y vs z (baseline vs optimized)
    - phase_maps.png              : baseline vs optimized element phase
    - after_lens_0p2um.mat/.zbf   : Ex, Ey 0.2 um past the optimized plate
                                    (zbf zero-padded to 512 = power of two, mm units)
    - optimized_R_map.csv         : asia_1310 realization for fabrication
    - summary.txt                 : metric table

Run from the repository root:
    python "Inversed design/Collimator_CircularFlat_Optimization.py"
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.io as sio
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
import optiprop
from zbf import ZbfBeam, PilotRays, write_zbf

OUTPUT_DIR = os.path.join(REPO_ROOT, 'output3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Slide-ready fonts
FS = 16
plt.rcParams.update({
    'font.size': FS, 'axes.titlesize': FS + 2, 'axes.labelsize': FS + 1,
    'xtick.labelsize': FS - 1, 'ytick.labelsize': FS - 1, 'legend.fontsize': FS - 3,
})


def add_colorbar(fig, ax, im, label):
    cax = make_axes_locatable(ax).append_axes('right', size='4.5%', pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 2)
    return cb


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
DESIGN_LAMBDA = 1.31e-6        # m
N_GLUE = 1.5
GLUE_DISTANCE = 40e-6          # m
LENS_DIAMETER = 90e-6          # m
PIXEL_SIZE = 325e-9            # m
FIELD_L = 160e-6               # m
PROP_AFTER_LENS = 0.2e-6       # m (export plane)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

W0_TARGET = 8e-6               # m, waist of the target circular Gaussian (at lens)
EVAL_Z = [150e-6, 400e-6, 800e-6]   # m, evaluation planes in air
GAMMA_FLAT = 1.0               # weight of the phase-flatness term
N_ITERS = 600
LR = 0.05
N_FFT = 1024                   # padded grid for the differentiable ASM (pow2)

SOURCE_DIR = os.path.join(REPO_ROOT, 'source')
_required = [f'P3_10um_{p}_{k}.txt' for p in ('EX', 'EY') for k in ('I', 'Phase')]
_missing = [f for f in _required if not os.path.exists(os.path.join(SOURCE_DIR, f))]
if _missing:
    sys.exit('Missing Zemax POP source files:\n  '
             + '\n  '.join(os.path.join('source', f) for f in _missing))

near_field = optiprop.NearField(pixel_size=PIXEL_SIZE, field_Lx=FIELD_L,
                                field_Ly=FIELD_L, device=DEVICE)
aperture = (near_field.X**2 + near_field.Y**2) <= (LENS_DIAMETER / 2)**2
Nx = int(near_field.Nx)

# ----------------------------------------------------------------------------
# Source fields at the metalens plane (after 40 um of glue)
# ----------------------------------------------------------------------------
def load_source(pol):
    src = optiprop.ZemaxPOPSource(near_field)
    src.calculate_phase(
        intensity_file=os.path.join(SOURCE_DIR, f'P3_10um_{pol}_I.txt'),
        phase_file=os.path.join(SOURCE_DIR, f'P3_10um_{pol}_Phase.txt'),
    )
    prop = optiprop.ASMPropagation(propagation_wavelength=DESIGN_LAMBDA,
                                   propagation_distance=GLUE_DISTANCE,
                                   n=N_GLUE, device=DEVICE)
    prop.set_input_field(u_in=src.U0, pixel_size=PIXEL_SIZE)
    prop.propagate()
    return prop.get_output_U


U_x = load_source('EX')
U_y = load_source('EY')

# ----------------------------------------------------------------------------
# Differentiable ASM: precomputed transfer functions on the padded grid
# ----------------------------------------------------------------------------
fx = torch.fft.fftfreq(N_FFT, d=PIXEL_SIZE, dtype=torch.float32, device=DEVICE)
FX, FY = torch.meshgrid(fx, fx, indexing='ij')
k_air = 2 * np.pi / DESIGN_LAMBDA
sqrt_term = torch.sqrt(1 - (DESIGN_LAMBDA**2) * (FX**2 + FY**2) + 0j)
H_planes = [torch.exp(1j * k_air * z * sqrt_term) for z in EVAL_Z]

pad0 = (N_FFT - Nx) // 2


def pad_field(U):
    return torch.nn.functional.pad(
        U, (pad0, N_FFT - Nx - pad0, pad0, N_FFT - Nx - pad0))


def propagate_planes(U):
    """Differentiable ASM to every evaluation plane. Returns padded fields."""
    Uf = torch.fft.fft2(pad_field(U))
    return [torch.fft.ifft2(Uf * H) for H in H_planes]


# Target: circular fundamental Gaussian MODE with waist w0 at the lens plane
# (flat phase at the waist; the known curvature/Gouy evolution downstream).
# Complex targets make the mode-purity term robust against the faint wide
# quantization halo (L2 inner product), unlike |sum U| / sum|U| style metrics.
x_pad_t = (torch.arange(N_FFT, dtype=torch.float32, device=DEVICE) - (N_FFT - 1) / 2) * PIXEL_SIZE
XXp, YYp = torch.meshgrid(x_pad_t, x_pad_t, indexing='xy')
R2p = XXp**2 + YYp**2
zR = np.pi * W0_TARGET**2 / DESIGN_LAMBDA
targets_amp = []       # normalized amplitude (for the shape loss / plots)
targets_mode = []      # normalized complex mode (for the purity loss)
for z in EVAL_Z:
    wz = W0_TARGET * float(np.sqrt(1 + (z / zR)**2))
    Rz = z * (1 + (zR / z)**2)
    gouy = float(np.arctan(z / zR))
    amp = torch.exp(-R2p / wz**2)
    phase = k_air * R2p / (2 * Rz) - gouy
    targets_amp.append(amp / torch.linalg.vector_norm(amp))
    mode = amp * torch.exp(1j * phase)
    targets_mode.append((mode / torch.linalg.vector_norm(mode)).to(torch.complex64))


# Incident power inside the aperture: the FIXED denominator of the coupling
# efficiency. Normalizing by the propagated power instead would let the
# optimizer cheat by dumping unwanted light into evanescent orders (the sharp
# 650 nm phase steps couple to spatial frequencies beyond 1/lambda, which
# decay within ~1 um and silently leave a "purer" surviving beam).
P_IN = (torch.abs(U_y * aperture) ** 2).sum()


def loss_terms(U_lens_out):
    """Returns (amplitude-shape loss, mode coupling loss) summed over planes.

    The mode term is 1 - |<T_k, U_k>|^2 / P_in: the fraction of the INCIDENT
    power coupled into the target Gaussian mode, so both wavefront mismatch
    and any power loss (absorption, evanescent scattering) are penalized.
    """
    L_amp = U_lens_out.new_zeros((), dtype=torch.float32)
    L_mode = U_lens_out.new_zeros((), dtype=torch.float32)
    for Uk, Ta, Tm in zip(propagate_planes(U_lens_out), targets_amp, targets_mode):
        a = torch.abs(Uk)
        a_n = a / torch.linalg.vector_norm(a)
        L_amp = L_amp + ((a_n - Ta) ** 2).sum()
        eta_c = torch.abs((torch.conj(Tm) * Uk).sum()) ** 2 / P_IN
        L_mode = L_mode + (1 - eta_c)
    return L_amp, L_mode


# ----------------------------------------------------------------------------
# Optimization (Ey only: 99.6 % of the power)
#
# v2: quantization-aware, fabrication-constrained.
#   - The phase is parameterized ON THE META-ATOM LATTICE (650 nm period,
#     one value per pillar) and nearest-upsampled to the simulation grid, so
#     the design contains no spatial detail a real lens cannot have. (A first
#     per-pixel attempt developed ~1.6 rad/pixel phase noise whose behavior
#     collapsed under any quantization.)
#   - The forward model quantizes the phase to the asia_1310 database
#     (nearest wrapped phase with the alpha amplitude penalty, including the
#     realized pillar amplitude) with a straight-through gradient estimator,
#     so the optimizer directly improves the REALIZED design.
#   - A small total-variation penalty on the lattice phase keeps the design
#     smooth in the low-intensity wings the loss barely sees.
# ----------------------------------------------------------------------------
ALPHA_DB = 0.3
LAMBDA_TV = 1e-6   # tiny: only suppresses noise in the low-intensity wings
N_LATTICE = (Nx + 1) // 2      # 650 nm lattice sites across the window

library = optiprop.MetaAtomLibrary(os.path.join(REPO_ROOT, 'data', 'metaatoms', 'asia_1310.npy'), device=DEVICE)
db_phase = library.phase.to(torch.float32)          # [30]
db_amp = library.amplitude.to(torch.float32)        # [30]
db_amp_penalty = ALPHA_DB * (1 - db_amp) ** 2


def wrap(x):
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


def upsample_to_grid(phi_lattice):
    """Nearest-neighbor upsample lattice phase (one pillar = 2x2 sim pixels)."""
    up = torch.repeat_interleave(torch.repeat_interleave(phi_lattice, 2, dim=0), 2, dim=1)
    return up[:Nx, :Nx]


def quantize_ste(phi_lattice):
    """Quantize the lattice phase to the database (one pillar = one entry),
    straight-through gradient, and upsample to the simulation grid."""
    with torch.no_grad():
        cost = wrap(db_phase.reshape(1, 1, -1) - phi_lattice.unsqueeze(-1)) ** 2 \
            + db_amp_penalty.reshape(1, 1, -1)
        idx = torch.argmin(cost, dim=-1)
        dphi = wrap(db_phase[idx] - phi_lattice)
        amp_lat = db_amp[idx]
    phase_q_lat = phi_lattice + dphi        # gradient flows through phi_lattice
    return upsample_to_grid(amp_lat), upsample_to_grid(phase_q_lat)


# Init: Ey phase conjugation sampled on the lattice (pillar centers)
phi_init_grid = (-torch.angle(U_y) * aperture).detach().to(torch.float32)
phi_lat = phi_init_grid[::2, ::2].clone()
if phi_lat.shape[0] != N_LATTICE:
    phi_lat = phi_lat[:N_LATTICE, :N_LATTICE].clone()
phi_lat.requires_grad_(True)
optimizer = torch.optim.Adam([phi_lat], lr=LR)

history = {'total': [], 'amp': [], 'mode': [], 'tv': []}
for it in range(N_ITERS):
    optimizer.zero_grad()
    amp_q, phase_q = quantize_ste(phi_lat)
    U_out = U_y * aperture * amp_q * torch.exp(1j * phase_q)
    L_amp, L_mode = loss_terms(U_out)
    tv = (wrap(phi_lat[1:, :] - phi_lat[:-1, :]) ** 2).sum() \
        + (wrap(phi_lat[:, 1:] - phi_lat[:, :-1]) ** 2).sum()
    loss = L_amp + GAMMA_FLAT * L_mode + LAMBDA_TV * tv
    loss.backward()
    optimizer.step()
    history['total'].append(loss.item())
    history['amp'].append(L_amp.item())
    history['mode'].append(L_mode.item())
    history['tv'].append(tv.item())
    if it % 50 == 0 or it == N_ITERS - 1:
        print(f'iter {it:4d}: total {loss.item():.5f} | amp {L_amp.item():.5f} '
              f'| mode {L_mode.item():.5f} | tv {LAMBDA_TV*tv.item():.5f}')

phi_opt = (upsample_to_grid(phi_lat.detach()) * aperture)
U0_baseline = aperture * torch.exp(-1j * torch.angle(U_y))         # conjugation
U0_optimized = aperture * torch.exp(1j * phi_opt)                  # continuous phase

# ----------------------------------------------------------------------------
# Realize the optimized phase with the asia_1310 database
# (optimize_global_offset=False so it matches the STE forward model exactly)
# ----------------------------------------------------------------------------
meta_lens = optiprop.MetaAtomElement(near_field)
meta_lens.calculate_phase(ideal_U0=U0_optimized, library=library,
                          alpha=ALPHA_DB, optimize_global_offset=False)
meta_lens.export_parameter_map(os.path.join(OUTPUT_DIR, 'optimized_R_map.csv'))

designs = {
    'conjugation (baseline)': U0_baseline,
    'optimized (ideal)': U0_optimized,
    'optimized (asia_1310)': meta_lens.U0,
}

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
x_pad_np = (np.arange(N_FFT) - (N_FFT - 1) / 2) * PIXEL_SIZE


def beam_sigma_np(I, x):
    Ix_, Iy_ = I.sum(axis=0), I.sum(axis=1)
    cx, cy = (Ix_*x).sum()/Ix_.sum(), (Iy_*x).sum()/Iy_.sum()
    return (np.sqrt((Ix_*(x-cx)**2).sum()/Ix_.sum()),
            np.sqrt((Iy_*(x-cy)**2).sum()/Iy_.sum()))


def mode_coupling(U, Tm):
    """|<T, U>|^2 / P_in: coupled fraction of the incident power."""
    return (torch.abs((torch.conj(Tm) * U).sum()) ** 2 / P_IN).item()


z_list = np.linspace(0, 1000e-6, 11)
metrics = {}
sigmas = {}
plane_amps = {}
with torch.no_grad():
    for name, L in designs.items():
        U_out = U_y * L
        # loss metrics at the eval planes
        Uks = propagate_planes(U_out)
        amp_mse = sum(((torch.abs(Uk)/torch.linalg.vector_norm(torch.abs(Uk)) - Ta)**2).sum().item()
                      for Uk, Ta in zip(Uks, targets_amp))
        eta_vals = [mode_coupling(Uk, Tm) for Uk, Tm in zip(Uks, targets_mode)]
        # far-field transmission: fraction of the incident power that survives
        # propagation (the rest went into evanescent orders / absorption)
        t_ff = ((torch.abs(Uks[0])**2).sum() / P_IN).item()
        circ = []
        for Uk in Uks:
            sx, sy = beam_sigma_np((torch.abs(Uk)**2).cpu().numpy(), x_pad_np)
            circ.append(sx / sy)
        plane_amps[name] = [(torch.abs(Uk)**2).cpu().numpy() for Uk in Uks]
        # sigma vs z
        Uf = torch.fft.fft2(pad_field(U_out))
        rows = []
        for z in z_list:
            if z == 0:
                Uz = pad_field(U_out)
            else:
                Hz = torch.exp(1j * k_air * float(z) * sqrt_term)
                Uz = torch.fft.ifft2(Uf * Hz)
            rows.append(beam_sigma_np((torch.abs(Uz)**2).cpu().numpy(), x_pad_np))
        sigmas[name] = np.array(rows)
        metrics[name] = {'amp_mse': amp_mse, 'eta': eta_vals, 'circ': circ, 't_ff': t_ff}
        print(f'{name:24s} ampMSE {amp_mse:.4f} | mode coupling '
              + '/'.join(f'{f:.3f}' for f in eta_vals)
              + f' | far-field transmission {t_ff:.3f}'
              + ' | circularity sx/sy ' + '/'.join(f'{c:.2f}' for c in circ)
              + f' | sigma@1mm {sigmas[name][-1,0]*1e6:.1f}x{sigmas[name][-1,1]*1e6:.1f} um')

# ----------------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------------
np.save(os.path.join(OUTPUT_DIR, 'optimized_phase.npy'), phi_opt.cpu().numpy())
sio.savemat(os.path.join(OUTPUT_DIR, 'optimized_phase.mat'), {
    'phase': phi_opt.cpu().numpy().astype(np.float64),
    'dx': PIXEL_SIZE, 'dy': PIXEL_SIZE, 'Nx': Nx, 'Ny': Nx,
    'lens_diameter': LENS_DIAMETER, 'design_lambda': DESIGN_LAMBDA,
})


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def save_zbf(path, Ex, Ey, wavelength_m, medium_index):
    N_ZBF = _next_pow2(Nx)
    Ex_np = optiprop.pad_to_center(Ex, N_ZBF).cpu().numpy().astype(np.complex128)
    Ey_np = optiprop.pad_to_center(Ey, N_ZBF).cpu().numpy().astype(np.complex128)
    x0 = (np.arange(Nx) - (Nx - 1)/2) * PIXEL_SIZE
    sx, sy = beam_sigma_np((torch.abs(Ey)**2).cpu().numpy(), x0)
    mm = 1e3
    pilot = PilotRays(posX=0.0, posY=0.0, waistX=sx*mm, waistY=sy*mm,
                      rayleighX=np.pi*sx**2*medium_index/wavelength_m*mm,
                      rayleighY=np.pi*sy**2*medium_index/wavelength_m*mm)
    write_zbf(ZbfBeam(nx=N_ZBF, ny=N_ZBF, dx=PIXEL_SIZE*mm, dy=PIXEL_SIZE*mm,
                      ispol=1, lam=wavelength_m*mm, Ex=Ex_np, Ey=Ey_np,
                      pilot=pilot, unit=0, index=medium_index), path)


with torch.no_grad():
    after = {}
    for pol, U in [('EX', U_x * meta_lens.U0), ('EY', U_y * meta_lens.U0)]:
        p = optiprop.ASMPropagation(propagation_wavelength=DESIGN_LAMBDA,
                                    propagation_distance=PROP_AFTER_LENS,
                                    n=1.0, device=DEVICE)
        p.set_input_field(u_in=U, pixel_size=PIXEL_SIZE)
        p.propagate()
        after[pol] = p.get_output_U
sio.savemat(os.path.join(OUTPUT_DIR, 'after_lens_0p2um.mat'), {
    'EX': after['EX'].cpu().numpy().astype(np.complex128),
    'EY': after['EY'].cpu().numpy().astype(np.complex128),
    'dx': PIXEL_SIZE, 'dy': PIXEL_SIZE, 'Nx': Nx, 'Ny': Nx,
})
save_zbf(os.path.join(OUTPUT_DIR, 'after_lens_0p2um.zbf'),
         after['EX'], after['EY'], DESIGN_LAMBDA, 1.0)

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
# Loss history
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.semilogy(history['total'], label='total')
ax.semilogy(history['amp'], label='amplitude shape')
ax.semilogy(history['mode'], label='mode impurity')
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax.set_title('Auto-gradient optimization history')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'optimization_loss.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# |Ey|^2 at eval planes: baseline vs optimized vs target
zoom = 60
fig, axes = plt.subplots(3, len(EVAL_Z), figsize=(6.2*len(EVAL_Z), 16))
fig.subplots_adjust(left=0.07, right=0.93, bottom=0.05, top=0.94, wspace=0.45, hspace=0.35)
ext_pad = [x_pad_np.min()*1e6, x_pad_np.max()*1e6, x_pad_np.min()*1e6, x_pad_np.max()*1e6]
for col, z in enumerate(EVAL_Z):
    for row, name in enumerate(['conjugation (baseline)', 'optimized (asia_1310)']):
        I = plane_amps[name][col]
        im = axes[row, col].imshow(I, extent=ext_pad, cmap='turbo', origin='lower')
        axes[row, col].set_title(f'{name.split(" (")[0]} |Ey|², z={z*1e6:.0f} µm')
        add_colorbar(fig, axes[row, col], im, 'Intensity')
    T2 = (targets_amp[col].cpu().numpy())**2
    im = axes[2, col].imshow(T2, extent=ext_pad, cmap='turbo', origin='lower')
    axes[2, col].set_title(f'target |T|², z={z*1e6:.0f} µm')
    add_colorbar(fig, axes[2, col], im, 'Intensity')
    for ax in axes[:, col]:
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
fig.savefig(os.path.join(OUTPUT_DIR, 'amp_vs_target.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# sigma vs z
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
styles = {'conjugation (baseline)': ('--s', 'tab:gray'),
          'optimized (ideal)': ('-^', 'tab:blue'),
          'optimized (asia_1310)': ('-o', 'tab:red')}
for ax, comp, lbl in [(axes[0], 0, '$\\sigma_x$'), (axes[1], 1, '$\\sigma_y$')]:
    for name, (st, c) in styles.items():
        ax.plot(z_list*1e6, sigmas[name][:, comp]*1e6, st, color=c, label=name)
    ax.set_xlabel('z in air after phase plate (µm)')
    ax.set_title(lbl)
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel('beam size $\\sigma$ (µm)')
fig.suptitle('Ey beam size vs z: conjugation baseline vs auto-gradient optimized')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'sigma_vs_z.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Phase maps
extent = [near_field.X.min().item()*1e6, near_field.X.max().item()*1e6,
          near_field.Y.min().item()*1e6, near_field.Y.max().item()*1e6]
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.88, wspace=0.45)
for ax, (name, U0) in zip(axes, [('conjugation (baseline)', U0_baseline),
                                 ('optimized (ideal)', U0_optimized),
                                 ('optimized (asia_1310)', meta_lens.U0)]):
    ph = (torch.angle(U0) % (2*np.pi)).cpu().numpy()
    im = ax.imshow(ph, extent=extent, cmap='turbo', origin='lower', vmin=0, vmax=2*np.pi)
    ax.set_title(name)
    add_colorbar(fig, ax, im, 'Phase (rad)')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
fig.savefig(os.path.join(OUTPUT_DIR, 'phase_maps.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Summary
lines = [f'Auto-gradient circular+flat beam shaping (Ey, target w0={W0_TARGET*1e6:.0f} um, '
         f'gamma={GAMMA_FLAT}, {N_ITERS} iters, lr={LR})',
         f'eval planes: {", ".join(f"{z*1e6:.0f}um" for z in EVAL_Z)}',
         '',
         'design                    ampMSE   modeCoupling@planes   ffTrans   circularity sx/sy      sigma@1mm (um)']
for name in designs:
    m = metrics[name]
    lines.append(f'{name:24s}  {m["amp_mse"]:.4f}   '
                 + '/'.join(f'{f:.3f}' for f in m['eta']) + '    '
                 f'{m["t_ff"]:.3f}   '
                 + '/'.join(f'{c:.2f}' for c in m['circ']) + '        '
                 f'{sigmas[name][-1,0]*1e6:.1f} x {sigmas[name][-1,1]*1e6:.1f}')
summary = '\n'.join(lines)
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
    f.write(summary + '\n')
print()
print(summary)
print('\nAll outputs saved to:', OUTPUT_DIR)
