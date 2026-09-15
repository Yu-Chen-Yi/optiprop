# Field Propagation Library

> Version 1.0.7 — organized project with moving-window ASM. Requires Python 3.10+.
> 功能盤點、移動視窗 API 與 GitHub 整理範圍：
> [CURRENT_FEATURES](docs/CURRENT_FEATURES.md)。
> 新範例：`python Example/ASM_MovingWindow.py`。
> 測試：先建立 `tmp/`，再執行 `python -m pytest tests --basetemp=tmp/test-run`。

A Python library for optical field propagation simulation, supporting multiple propagation methods and optical element modeling.

## Features

- 🔬 **Rich optical element models**: lenses, diffractive optical elements, VCSEL, etc.
- 🌊 **Multiple propagation algorithms**: Fresnel, Angular Spectrum, Rayleigh–Sommerfeld integral
- 🚀 **GPU acceleration**: built on PyTorch with CUDA support
- 📊 **Visualization tools**: convenient plotting utilities built in
- 🎯 **Easy to use**: clean API suitable for research and teaching

## Installation

### From source

```bash
git clone https://github.com/Yu-Chen-Yi/optiprop.git
cd optiprop
pip install -e .
```

For the PySide6 near-field workbench:

```bash
pip install -e ".[gui]"
optiprop-gui
```

The desktop application builds an incident field, lets you arrange and edit
an arbitrary layer stack, runs the propagation outside the UI thread, and
shows amplitude, wrapped phase, intensity, line profiles, and XZ scans.
Projects use a versioned JSON document; imported `.npz`, `.mat`, and `.zbf`
assets remain separate and are checked by SHA-256 when the project is opened.

Two non-GUI tools are installed as well:

```bash
optiprop-inspect input.zbf --json
optiprop-convert input.zbf output.npz
```

### Via pip (recommended)

```bash
pip install --index-url https://test.pypi.org/simple/ optiprop
```

### GPU support

If you need GPU acceleration, install the CUDA build of PyTorch

## Quick Start

### Canonical field model (2.0 migration foundation)

New multilayer and GUI code should use the canonical `[component, y, x]`
field contract.  Existing `NearField` and propagator APIs remain available
during the migration.

```python
import torch
import optiprop

grid = optiprop.Grid2D(
    nx=192,
    ny=127,
    dx=0.37e-6,
    dy=0.81e-6,
)
field = optiprop.Field2D(
    data=torch.ones((1, grid.ny, grid.nx), dtype=torch.complex64),
    grid=grid,
    wavelength_m=1.31e-6,
)

X, Y = grid.meshgrid()
print(field.intensity().shape)  # [ny, nx]

spec = optiprop.PropagationSpec(
    method="blas",
    distance_m=1e-3,
    padding=optiprop.PaddingSpec.auto(),
)
preflight = optiprop.assess_sampling(field, spec)
print(preflight.padded_shape, [issue.code for issue in preflight.issues])

result = optiprop.propagate_angular_spectrum(field, spec)
print(result.field.shape, result.power_after)
print(result.diagnostics["bandlimit"])
```

The canonical backend supports same-grid ASM and coupled band-limited ASM
(`BLAS`), rectangular grids, scalar or `Ex/Ey` fields, CPU/CUDA,
`complex64`/`complex128`, zero padding with inverse crop, and structured
sampling diagnostics. It uses the `exp(-iωt)` convention, so a positive-Z
plane wave advances with `exp(+ikz z)`. Passive material loss is represented
by `Im(n) >= 0`; gain media and unstable backward propagation through loss
are rejected during preflight. Evanescent `discard` is the default, `decay`
provides stable regularized propagation in either Z direction for real
indices, and advanced `keep` performs exact continuation with conditioning
warnings or overflow protection.

Canonical Fresnel propagation offers two paths:

```python
# Same-grid paraxial transfer function; padding is cropped back to input.
tf = optiprop.propagate_fresnel(
    field,
    optiprop.PropagationSpec(
        method="fresnel_tf",
        distance_m=2e-3,
        padding=optiprop.PaddingSpec.auto(),
    ),
)

# Single-FFT Fresnel; the returned grid has natural DFT output sampling.
scaled = optiprop.propagate_fresnel(
    field,
    optiprop.PropagationSpec(
        method="fresnel_scaled",
        distance_m=20e-3,
        padding=optiprop.PaddingSpec.none(),
    ),
)
print(scaled.field.grid.dx, scaled.field.grid.dy)
```

`fresnel_scaled` supports rectangular and odd/even grids, independent X/Y
sampling, shifted observation-window centres, and positive or negative
distance. Its output sampling is determined by wavelength, distance, FFT
shape, and input spacing; incompatible custom sampling is rejected during
preflight.

Canonical Rayleigh–Sommerfeld propagation uses the complete RS-I near-field
kernel:

```python
# Same-grid exact RS-I linear convolution. Auto padding prevents wraparound.
rs_fft = optiprop.propagate_rayleigh_sommerfeld(
    field,
    optiprop.PropagationSpec(
        method="rs_fft",
        distance_m=50e-6,
        padding=optiprop.PaddingSpec.auto(),
    ),
)

# Arbitrary regular output grid, evaluated in memory-bounded chunks.
observation = optiprop.Grid2D(
    nx=101,
    ny=81,
    dx=0.5e-6,
    dy=0.7e-6,
    x_center=10e-6,
)
rs_direct = optiprop.propagate_rayleigh_sommerfeld(
    field,
    optiprop.PropagationSpec(
        method="rs_direct",
        distance_m=20e-6,
        output_grid=observation,
        options={"direct_memory_budget_bytes": 128 * 1024**2},
    ),
)
```

`rs_fft` requires a linear-convolution buffer of at least
`(2*ny-1, 2*nx-1)`. `rs_direct` automatically selects a chunk size from its
memory budget, or accepts an explicit `options={"chunk_size": ...}`.
Canonical RS-I currently supports forward distance only; use ASM with an
explicit evanescent policy when numerical backpropagation is required.

### Ordered multilayer systems

An `IncidentSource` creates the canonical input field; it is deliberately
separate from the field-in/field-out layer list. Propagation, lens, aperture,
complex-mask, and interface operations can then be composed into an immutable
optical system. Every layer has a stable UUID suitable for a Qt list model,
and every run retains its input plus per-layer output snapshots:

```python
import torch
import optiprop

grid = optiprop.Grid2D(
    nx=256,
    ny=192,
    dx=1e-6,
    dy=1e-6,
)
source = optiprop.IncidentSource.gaussian(
    grid,
    wavelength_m=940e-9,
    waist_x_m=60e-6,       # 1/e field-amplitude radius
    amplitudes=(1.0,),
    phase_offset_rad=0.0,
)
field = source.create(dtype=torch.complex128)

system = optiprop.OpticalSystem(
    name="Near-field stack",
    layers=(
        optiprop.PropagationLayer(
            name="Source to lens",
            spec=optiprop.PropagationSpec(
                method="asm",
                distance_m=20e-6,
                medium_index=1.0,
                padding=optiprop.PaddingSpec.auto(),
            ),
        ),
        optiprop.IdealLensLayer(
            name="Ideal lens",
            focal_length_m=500e-6,
            design_wavelength_m=940e-9,
            transmission_amplitude=0.95,
            aperture=optiprop.ApertureSpec(
                shape="circle",
                size_x_m=160e-6,
            ),
        ),
        optiprop.InterfaceLayer(
            name="Air to glass",
            n1=1.0,
            n2=1.45,
        ),
        optiprop.PropagationLayer(
            name="Glass layer",
            spec=optiprop.PropagationSpec(
                method="blas",
                distance_m=100e-6,
                medium_index=1.45,
                padding=optiprop.PaddingSpec.auto(),
            ),
        ),
        optiprop.InterfaceLayer(
            name="Glass to air",
            n1=1.45,
            n2=1.0,
        ),
        optiprop.PropagationLayer(
            name="Air to observation",
            spec=optiprop.PropagationSpec(
                method="fresnel_tf",
                distance_m=500e-6,
                medium_index=1.0,
                padding=optiprop.PaddingSpec.auto(),
            ),
        ),
    ),
)

run = system.execute(field)
for layer_result in run.layer_results:
    print(
        layer_result.layer_name,
        layer_result.status.value,
        layer_result.output_field.z_m,
    )
```

Layer edits such as `append`, `insert`, `remove`, `move`, `update`, and
`set_enabled` return a new `OpticalSystem`. Disabled layers are exact identity
operations with a `SKIPPED` result. Validation errors are addressed by layer
UUID and property path; cancellation or backend failure carries completed
partial results for UI recovery.

`IdealLensLayer` defaults to an exact equal-path phase and also supports the
explicit `LensPhaseModel.PARAXIAL` model. `ComplexMaskLayer` accepts a scalar
mask or one transmission map per Ex/Ey component; grid mismatch is rejected
unless `MaskSamplingMode.RESAMPLE_COMPLEX` is explicitly selected.

An index change inside a `PropagationLayer` is rejected: add an
`InterfaceLayer` first so the medium transition is visible in the layer list.
The current interface model applies a normal-incidence scalar Fresnel
coefficient and discards the reflected branch; it does not model oblique s/p
coefficients, Fabry–Perot interference, or bidirectional propagation.
`Field2D.power()` remains integrated relative intensity and is not
impedance-corrected physical power across an index change.

### Canonical NPZ and MAT field I/O

Canonical NPZ and MATLAB v5 files use the same explicit `Field2D` schema,
including `[C,Y,X]` axis order, components, SI grid spacing, wavelength,
complex medium index, plane position, conventions, and JSON metadata:

```python
report = optiprop.save_field(run.output_field, "output.npz")
inspection = optiprop.inspect_field("output.npz")
restored = optiprop.load_field("output.npz")

print(report.sha256)
print(inspection.is_canonical, inspection.array("field").shape)
```

Noncanonical files are inspected but never silently loaded. Legacy aliases
and units must be confirmed with an explicit, reusable mapping:

```python
mapping = optiprop.ImportMapping(
    data_keys=("EX", "EY"),
    components=("Ex", "Ey"),
    dx_key="dx",
    dy_key="dy",
    spatial_unit=optiprop.LengthUnit.M,
    # Override known-bad or missing legacy wavelength metadata explicitly.
    wavelength_m=1.31e-6,
    medium_index=1.0,
    axis_order=optiprop.AxisOrder.YX,
)
legacy_field = optiprop.load_field("legacy.mat", mapping)
```

NPZ loading always uses `allow_pickle=False` and rejects object arrays,
duplicate/path-traversal ZIP members, excessive compression ratios, malformed
schemas, and non-finite fields. Saves are atomic by default. MAT v4/v5 through
v7.2 use SciPy; MATLAB v7.3/HDF5 currently returns an explicit unsupported-
format error instead of applying an unverified transpose.

### Basic example

```python
import optiprop
import torch

# Create a near-field grid
field = optiprop.NearField(
    pixel_size=1e-6,      # pixel size 1 μm
    field_Lx=1000e-6,     # field length 1 mm
    field_Ly=1000e-6,     # field width 1 mm
    device='cpu'          # compute on CPU
)

# Create an equal optical path lens
lens = optiprop.EqualPathPhase(field)
U0 = lens.calculate_phase(
    focal_length=1000e-6,     # focal length 1 mm
    design_lambda=0.94e-6,    # design wavelength 940 nm
    lens_diameter=800e-6      # lens diameter 800 μm
)

# Visualize lens phase
lens.draw(selected_field='phase')

# Create a propagator
propagator = optiprop.FresnelPropagation(
    propagation_wavelength=0.94e-6,  # wavelength
    propagation_distance=1000e-6,    # distance 1 mm
    device='cpu'
)

# Set input field
propagator.set_input_field(U0, field.pixel_size)

# Run propagation
propagator.propagate()

# Show result
propagator.show_intensity(title='Focal intensity distribution')
```

### Advanced example

```python
# Create a diffractive optics element
doe = optiprop.DiffractiveOpticsElement(field)

# Define a unit cell pattern
unit_cell = torch.rand(50, 50)  # random phase pattern
U_doe = doe.calculate_phase(
    unit_cell=unit_cell,
    xperiod=5e-6,
    yperiod=5e-6
)

# Propagate using the Angular Spectrum Method
asm_prop = optiprop.ASMPropagation(
    propagation_wavelength=0.94e-6,
    propagation_distance=2000e-6
)
asm_prop.set_input_field(U_doe, field.pixel_size)
asm_prop.propagate()

# Visualize propagation result
asm_prop.show_intensity()
```

## Supported Optical Elements

| Element | Class | Description |
|---------|-------|-------------|
| Equal optical path lens | `EqualPathPhase` | Standard spherical lens |
| Cubic phase element | `CubicPhase` | Extended depth-of-field lens |
| Binary phase element | `Binary2Phase` | Polynomial phase distribution |
| Diffractive optics element | `DiffractiveOpticsElement` | Periodic structures |
| VCSEL source | `VSCELPhase` | Gaussian beam source |
| Incident field | `IncidentField` | Plane wave and tilted incidence |
| Zemax POP source | `ZemaxPOPSource` | Load POP irradiance/phase text listings |
| Meta-atom element | `MetaAtomElement` | Realize an ideal phase with a meta-atom database |

## Meta-Atom Database Support

Replace an ideal phase profile with the realizable amplitude/phase of
fabricated meta-atoms (nearest wrapped-phase lookup):

```python
library = optiprop.MetaAtomLibrary('data/metaatoms/asia_1310.npy')   # .npy with a 'data_sheet' dict
library.rich_print()                                  # phase coverage statistics

meta_lens = optiprop.MetaAtomElement(field)
meta_lens.calculate_phase(
    ideal_element=ideal_lens,        # any PhaseElement (or ideal_U0=...)
    library=library,
    alpha=0.3,                       # amplitude penalty: avoid lossy resonances
    optimize_global_offset=True,     # steer away from phase-coverage gaps
)
meta_lens.export_parameter_map('R_map.csv')  # structure parameter per pixel
U_out = incident.U0 * meta_lens.U0           # propagate as usual
```

## Supported Propagation Methods

| Method | Class | Applicable Range |
|--------|-------|------------------|
| Fresnel propagation | `FresnelPropagation` | Near to far field |
| Angular Spectrum | `ASMPropagation` | Any distance, same grid |
| Rayleigh–Sommerfeld | `RayleighSommerfeldPropagation` | Any distance, arbitrary grid |

## API Reference

### Core classes

#### NearField
Create computation grid and coordinates.

```python
field = optprop.NearField(
    pixel_size=1e-6,        # pixel size (m)
    field_Lx=1000e-6,       # field size in X (m)
    field_Ly=1000e-6,       # field size in Y (m)
    dtype=torch.float32,    # data type
    field_center=[0, 0],    # field center (m)
    device='cpu'            # compute device
)
```

#### PhaseElement (abstract base class)
Base class for all optical elements, providing a common interface.

Key methods:
- `calculate_phase(**kwargs)`: compute phase distribution
- `draw()`: plot field distributions
- `rich_print()`: display element parameters

#### Propagation classes
All propagation classes share a similar interface:

```python
# Create propagator
prop = optprop.FresnelPropagation(wavelength, distance, device='cpu')

# Set input field
prop.set_input_field(input_field, pixel_size)

# Run propagation
prop.propagate()

# Show result
prop.show_intensity()
```

## Performance Tips

### GPU acceleration
```python
# Use GPU acceleration
field = optprop.NearField(device='cuda')
prop = optprop.FresnelPropagation(device='cuda')
```

### Memory management
```python
# For large-scale computations, choose appropriate dtypes
field = optprop.NearField(dtype=torch.float32)  # save memory
# field = optprop.NearField(dtype=torch.float64)  # higher precision
```

## Examples and Tutorials

See the `Example/` directory for more examples:

- `Metalens_Example1_ASM/FS/RS`: metalens focusing with the three propagators
- `Metalens_Example2_MetaAtom`: realistic metalens from the bundled `asia_1310.npy` meta-atom library
- `Dot_projector_Example1`: diffractive dot projector
- `hopfion`: hopfion field synthesis
- `Laser2Metalens_Collimator_MetaAtom` / `Laser2Metalens_Collimator_IdealPhase`:
  laser collimator from Zemax POP sources (requires POP `.txt` exports under
  `source/`, not distributed with the repository due to file size)
- `Laser2Metalens_Collimator_IdealPhase.oprop`: ready-to-open GUI version of
  the ideal-phase collimator. It imports the generated
  `collimator_ideal_before_lens.mat`, applies the 16 um hyperbolic lens,
  90 um aperture, glue-to-air interface, 0.2 um ASM propagation, and a
  201-plane XZ scan. Launch it with:

  ```bash
  optiprop-gui Example/Laser2Metalens_Collimator_IdealPhase.oprop
  ```

### Google Colab Examples

- Metalens_Example1_ASM: [Open in Colab](https://colab.research.google.com/drive/1Ma4orvFWycHQf_wHgouamfz4lzwbHBni?usp=sharing)
- Metalens_Example1_FS: [Open in Colab](https://colab.research.google.com/drive/1ftKEIuFW7REQUulPLxjgtFcksQSKo9g0?usp=sharing)
- Metalens_Example1_RS: [Open in Colab](https://colab.research.google.com/drive/1uf8jjfxTqPnOvoxkKKCNfnZq9aXiBZwb?usp=sharing)
- Dot_projector_Example1.ipynb: [Open in Colab](https://colab.research.google.com/drive/12UWHKb53WILAiJ-OntZEBm3BoDTecmv7?usp=sharing)
- Hopfion: [Open in Colab](https://colab.research.google.com/drive/1p0D2fgxaKTfDtEx0W-IKAFmX-5Qs8bVQ?usp=sharing)
## Dependencies

- Python >= 3.10
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- scikit-image >= 0.18.0
- SciPy >= 1.7.0
- Rich >= 10.0.0

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development setup

```bash
git clone https://github.com/Yu-Chen-Yi/optiprop.git
cd optiprop
pip install -e .[dev]
```

### Run tests

```bash
python test_basic.py
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{optiprop,
  title={Field Propagation Library},
  author={Yu-Chen-Yi},
  year={2025},
  url={https://github.com/Yu-Chen-Yi/optiprop}
}
```

## Contact

- Author: Yu-Chen-Yi
- Email: chenyi@g.ncu.edu.tw
- GitHub Issues: [Report an issue](https://github.com/Yu-Chen-Yi/optiprop/issues)
- Docs: [Online documentation](https://optiprop.readthedocs.io/)

## Changelog

### v1.0.7

- Add parallel moving-window ASM with fractional X/Y shifts and global-coordinate XZ scans; see `Example/ASM_MovingWindow.py`.
- Include canonical Field2D/Grid2D, polarization-aware I/O, optical systems, project persistence, CLI and optional GUI.
- Organize the example meta-atom libraries under `data/metaatoms/` (available in the repository and source distribution).
- Keep legacy propagation interfaces; Python 3.10+ is now required. GUI needs `pip install 'optiprop[gui]'`.
- Exclude private source fields, generated simulation results and caches from distribution.
- Moving windows are not tilted planes or shifted BLAS; check padding and sampling convergence.

### v1.0.0 (2025-10-07)
- Initial release
- Basic optical elements and propagation algorithms
- GPU acceleration support
- Complete docs and examples

### v1.0.4 (2025-10-28)
- Fix plot figure bug
- Add figure option

### v1.0.6 (2026-07-04)
- **Meta-atom database support** (`optiprop.metaatom`):
  - `MetaAtomLibrary` — load a meta-atom database (`.npy` with a `data_sheet`
    dict), nearest wrapped-phase lookup with optional amplitude penalty
    (`alpha`), phase-coverage statistics, `rich_print()` / `plot()`
  - `MetaAtomElement` — replace an ideal `PhaseElement` field with realizable
    meta-atom amplitude/phase, optional global phase-offset optimization,
    phase-error maps, and `export_parameter_map()` (R per pixel) for fabrication
- **Zemax POP source** (`optiprop.ZemaxPOPSource`) — load "Listing of POP
  Irradiance/Phase Data" text exports (UTF-16), reconstruct
  `U = sqrt(I)*exp(j*phase)` and bilinearly resample onto the simulation grid
- New examples: `Metalens_Example2_MetaAtom` and
  `Laser2Metalens_Collimator_*` (single-phase dual-polarization collimator)
