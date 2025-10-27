# Field Propagation Library

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

### Via pip (recommended)

```bash
pip install --index-url https://test.pypi.org/simple/ optiprop
```

### GPU support

If you need GPU acceleration, install the CUDA build of PyTorch

## Quick Start

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

See the `examples/` directory for more examples:

- `basic_lens_simulation.py`: basic lens simulation
- `doe_design.py`: diffractive optics design
- `beam_propagation.py`: beam propagation analysis
- `gpu_acceleration.py`: GPU acceleration

### Google Colab Examples

- Metalens_Example1_ASM: [Open in Colab](https://colab.research.google.com/drive/1Ma4orvFWycHQf_wHgouamfz4lzwbHBni?usp=sharing)
- Metalens_Example1_FS: [Open in Colab](https://colab.research.google.com/drive/1ftKEIuFW7REQUulPLxjgtFcksQSKo9g0?usp=sharing)
- Metalens_Example1_RS: [Open in Colab](https://colab.research.google.com/drive/1uf8jjfxTqPnOvoxkKKCNfnZq9aXiBZwb?usp=sharing)
- Dot_projector_Example1.ipynb: [Open in Colab](https://colab.research.google.com/drive/12UWHKb53WILAiJ-OntZEBm3BoDTecmv7?usp=sharing)
- Hopfion: [Open in Colab](https://colab.research.google.com/drive/1p0D2fgxaKTfDtEx0W-IKAFmX-5Qs8bVQ?usp=sharing)
## Dependencies

- Python >= 3.7
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
pytest tests/
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

### v1.0.0 (2025-10-07)
- Initial release
- Basic optical elements and propagation algorithms
- GPU acceleration support
- Complete docs and examples

### v1.0.4 (2025-10-28)
- Fix plot figure bug
- Add figure option