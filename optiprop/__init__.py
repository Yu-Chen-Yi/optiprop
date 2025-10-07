"""
Field Propagation Library

A Python library for optical field propagation simulation, supporting multiple propagation methods and optical elements.

Main Features:
- Optical element modeling (lenses, diffractive optical elements, etc.)
- Multiple propagation algorithms (Fresnel, Angular Spectrum, Rayleigh-Sommerfeld)
- Visualization tools
- GPU acceleration support

Author: Yu-Chen-Yi
Email: chenyi@g.ncu.edu.tw
Version: 1.0.0
"""

# Import main classes and functions
from .elements import (
    NearField,
    PhaseElement,
    Binary2Phase,
    EqualPathPhase,
    CubicPhase,
    Binary1Phase,
    VSCELPhase,
    DiffractiveOpticsElement,
    IncidentField
)

from .propagation import (
    FresnelPropagation,
    ASMPropagation,
    RayleighSommerfeldPropagation
)

from .utils import (
    cart_grid,
    get_grid_size,
    pad_to_center,
    check_available_cuda,
    extract_peak_points,
    plot_field_amplitude_phase,
    plot_field_intensity,
    plot_xz_field_intensity
)

# Version information
__version__ = "1.0.0"
__author__ = "Yu-Chen-Yi"
__email__ = "chenyi@g.ncu.edu.tw"

# Define public API
__all__ = [
    # Optical elements
    'NearField',
    'PhaseElement',
    'Binary2Phase',
    'EqualPathPhase',
    'CubicPhase',
    'Binary1Phase',
    'VSCELPhase',
    'DiffractiveOpticsElement',
    'IncidentField',
    
    # Propagation algorithms
    'FresnelPropagation',
    'ASMPropagation',
    'RayleighSommerfeldPropagation',
    
    # Utility functions
    'cart_grid',
    'get_grid_size',
    'pad_to_center',
    'check_available_cuda',
    'extract_peak_points',
    'plot_field_amplitude_phase',
    'plot_field_intensity',
    'plot_xz_field_intensity',
]

# Package-level configuration
import warnings

def configure_warnings():
    """Configure warning messages"""
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Configure on initialization
configure_warnings()

# Package information
def info():
    """Display package information"""
    print(f"Field Propagation Library v{__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print("A Python library for optical field propagation simulation")
    print("\nMain modules:")
    print("- elements: Optical element modeling")
    print("- propagation: Propagation algorithms")
    print("- utils: Utility functions")
    print("\nUsage example:")
    print("import optiprop")
    print("field = optiprop.NearField(pixel_size=1e-6)")
    print("lens = optiprop.EqualPathPhase(field)")
    print("prop = optiprop.FresnelPropagation()")
