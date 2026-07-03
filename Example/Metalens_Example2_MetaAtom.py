"""
Metalens Example 2: Realistic metalens from a meta-atom database.

Workflow:
    1. Build an ideal EqualPathPhase metalens (f = 500 um, lambda = 1.31 um).
    2. Load the meta-atom library (asia_1310.npy, circular Si pillars,
       period 650 nm) and replace the ideal phase/amplitude with the nearest
       realizable meta-atom values (MetaAtomElement).
    3. Propagate BOTH the ideal and the realized field to the focal plane
       with the Angular Spectrum Method and compare the focal peaks.
    4. Export the structure parameter map (R per pixel) for fabrication.

Run from the repository root:
    python Example/Metalens_Example2_MetaAtom.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')  # no blocking windows
import torch

# Allow running the example without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import optiprop

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LIBRARY_PATH = os.path.join(REPO_ROOT, 'asia_1310.npy')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Design parameters
# ----------------------------------------------------------------------------
DESIGN_LAMBDA = 1.31e-6      # m
FOCAL_LENGTH = 500e-6        # m
LENS_DIAMETER = 180e-6       # m
PIXEL_SIZE = 650e-9          # m (= meta-atom lattice period of the database)
FIELD_L = 200e-6             # m
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------------------
# 1. Near field and ideal metalens
# ----------------------------------------------------------------------------
near_field = optiprop.NearField(
    pixel_size=PIXEL_SIZE,
    field_Lx=FIELD_L,
    field_Ly=FIELD_L,
    device=DEVICE,
)

ideal_lens = optiprop.EqualPathPhase(near_field)
ideal_lens.calculate_phase(
    focal_length=FOCAL_LENGTH,
    design_lambda=DESIGN_LAMBDA,
    lens_diameter=LENS_DIAMETER,
    aperture_type='circle',
    aperture_size=[LENS_DIAMETER],
)

# ----------------------------------------------------------------------------
# 2. Meta-atom library and realized metalens
# ----------------------------------------------------------------------------
library = optiprop.MetaAtomLibrary(LIBRARY_PATH, device=DEVICE)
library.rich_print()
library.plot(show=False, save_path=os.path.join(OUTPUT_DIR, 'metaatom_library.png'))

meta_lens = optiprop.MetaAtomElement(near_field)
meta_lens.calculate_phase(
    ideal_element=ideal_lens,
    library=library,
    alpha=0.3,
    optimize_global_offset=True,
)
meta_lens.rich_print()

# Fabrication export and inspection figures
meta_lens.export_parameter_map(os.path.join(OUTPUT_DIR, 'metalens_R_map.csv'))
meta_lens.draw_parameter_map(show=False, save_path=os.path.join(OUTPUT_DIR, 'metalens_R_map.png'))
meta_lens.draw_phase_error(show=False, save_path=os.path.join(OUTPUT_DIR, 'metalens_phase_error.png'))

# ----------------------------------------------------------------------------
# 3. Propagate ideal and realized lenses to the focal plane (ASM)
# ----------------------------------------------------------------------------
incident = optiprop.IncidentField(near_field)
incident.calculate_phase(
    incident_angle=[0, 0],
    design_lambda=DESIGN_LAMBDA,
    aperture_type='rectangle',
    aperture_size=[FIELD_L, FIELD_L],
)

peaks = {}
for name, lens in [('ideal', ideal_lens), ('metaatom', meta_lens)]:
    prop = optiprop.ASMPropagation(
        propagation_wavelength=DESIGN_LAMBDA,
        propagation_distance=FOCAL_LENGTH,
        device=DEVICE,
    )
    prop.set_input_field(u_in=incident.U0 * lens.U0, pixel_size=PIXEL_SIZE)
    prop.propagate()
    U_focal = prop.get_output_U
    intensity = torch.abs(U_focal) ** 2
    peaks[name] = intensity.max().item()
    print(f"{name:8s} focal peak intensity: {peaks[name]:.4e}")

    optiprop.plot_field_intensity(
        U_focal, near_field.X, near_field.Y,
        xlim=[-20, 20], ylim=[-20, 20],
        title=f"Focal plane ({name})",
        show=False,
    )
    matplotlib.pyplot.savefig(
        os.path.join(OUTPUT_DIR, f'metalens_focal_{name}.png'),
        dpi=300, bbox_inches='tight',
    )
    matplotlib.pyplot.close('all')

ratio = peaks['metaatom'] / peaks['ideal']
print(f"\nRealized/ideal focal peak intensity ratio: {ratio:.4f}")

# ----------------------------------------------------------------------------
# 4. Sanity checks
# ----------------------------------------------------------------------------
aperture = meta_lens.aperture
R_in = meta_lens.parameter_map[aperture]
R_out = meta_lens.parameter_map[~aperture]
err = meta_lens.phase_error[aperture]
print(f"R map inside aperture: {R_in.min().item():.1f} ~ {R_in.max().item():.1f} nm "
      f"({int(aperture.sum())} pixels)")
print(f"R map outside aperture (should be 0): max = {R_out.abs().max().item():.1f}")
print(f"Phase error (alpha=0.3): mean |err| = {err.abs().mean().item():.4f} rad, "
      f"max |err| = {err.abs().max().item():.4f} rad")
print("Figures and R map saved to:", OUTPUT_DIR)
