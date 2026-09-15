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
Version: 1.0.7
"""

# Import main classes and functions
from .core import (
    EX_EY_COMPONENTS,
    SCALAR_COMPONENTS,
    Field2D,
    Grid2D,
    Severity,
    ValidationError,
    ValidationIssue,
    ValidationReport,
)

from .elements import (
    NearField,
    PhaseElement,
    Binary2Phase,
    EqualPathPhase,
    CubicPhase,
    AxiCubicPhase,
    Binary1Phase,
    VSCELPhase,
    DiffractiveOpticsElement,
    IncidentField,
    GaussianBeamSource,
    ZemaxPOPSource
)

from .metaatom import (
    MetaAtomLibrary,
    MetaAtomElement
)

from .propagation import (
    AngularSpectrumPropagator,
    CancellationToken,
    EvanescentPolicy,
    FresnelPropagation,
    FresnelPropagator,
    ASMPropagation,
    PaddingMode,
    PaddingSpec,
    PrecisionPolicy,
    PropagationCancelled,
    PropagationMethod,
    PropagationResult,
    PropagationSpec,
    Propagator,
    RayleighSommerfeldPropagation,
    RayleighSommerfeldPropagator,
    SamplingReport,
    assess_sampling,
    propagate_angular_spectrum,
    propagate_fresnel,
    propagate_rayleigh_sommerfeld,
)

from .system import (
    ApertureLayer,
    ApertureShape,
    ApertureSpec,
    ComplexMaskLayer,
    FieldGeometry,
    IdealLensLayer,
    IncidentSource,
    InterfaceLayer,
    InterfaceModel,
    LensPhaseModel,
    LayerBase,
    LayerExecutionStatus,
    LayerMetadata,
    LayerResult,
    OpticalLayer,
    OpticalSystem,
    OpticalSystemResult,
    PropagationBackendRegistry,
    PropagationLayer,
    RunContext,
    MaskSamplingMode,
    SourceKind,
    SystemExecutionCancelled,
    SystemExecutionFailed,
    SystemValidationError,
    ThinElementLayer,
    default_backend_registry,
)

from .io import (
    AmbiguousMappingError,
    AmplitudeConvention,
    ArrayInspection,
    AxisOrder,
    DatasetInspection,
    ExportOptions,
    ExportReport,
    FieldFormat,
    FieldIOError,
    ImportMapping,
    ImportMappingError,
    ImportResult,
    LengthUnit,
    MappingConfidence,
    PhaseUnit,
    SchemaValidationError,
    UnsafeDatasetError,
    UnsupportedFormatError,
    ZbfBeam,
    ZbfPilotRays,
    inspect_field,
    inspect_zbf,
    load_field,
    load_zbf,
    read_zbf,
    save_field,
    save_zbf,
    write_zbf,
)

from .project import (
    AssetIntegrityError,
    AssetReference,
    AssetStatus,
    ComputeSettings,
    OptiPropProject,
    ProjectError,
    load_project,
    project_to_domain,
    save_project,
    validate_assets,
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
__version__ = "1.0.7"
__author__ = "Yu-Chen-Yi"
__email__ = "chenyi@g.ncu.edu.tw"

# Define public API
__all__ = [
    # Canonical OptiProp 2.0 field model
    'Grid2D',
    'Field2D',
    'SCALAR_COMPONENTS',
    'EX_EY_COMPONENTS',
    'Severity',
    'ValidationError',
    'ValidationIssue',
    'ValidationReport',

    # Optical elements
    'NearField',
    'PhaseElement',
    'Binary2Phase',
    'EqualPathPhase',
    'CubicPhase',
    'AxiCubicPhase',
    'Binary1Phase',
    'VSCELPhase',
    'DiffractiveOpticsElement',
    'IncidentField',
    'GaussianBeamSource',
    'ZemaxPOPSource',

    # Meta-atom database
    'MetaAtomLibrary',
    'MetaAtomElement',

    # Propagation algorithms
    'FresnelPropagation',
    'FresnelPropagator',
    'ASMPropagation',
    'AngularSpectrumPropagator',
    'RayleighSommerfeldPropagation',
    'RayleighSommerfeldPropagator',
    'CancellationToken',
    'EvanescentPolicy',
    'PaddingMode',
    'PaddingSpec',
    'PrecisionPolicy',
    'PropagationCancelled',
    'PropagationMethod',
    'PropagationResult',
    'PropagationSpec',
    'Propagator',
    'SamplingReport',
    'assess_sampling',
    'propagate_angular_spectrum',
    'propagate_fresnel',
    'propagate_rayleigh_sommerfeld',

    # Immutable multilayer optical system
    'ApertureLayer',
    'ApertureShape',
    'ApertureSpec',
    'ComplexMaskLayer',
    'FieldGeometry',
    'IdealLensLayer',
    'IncidentSource',
    'InterfaceLayer',
    'InterfaceModel',
    'LensPhaseModel',
    'LayerBase',
    'LayerExecutionStatus',
    'LayerMetadata',
    'LayerResult',
    'OpticalLayer',
    'OpticalSystem',
    'OpticalSystemResult',
    'PropagationBackendRegistry',
    'PropagationLayer',
    'RunContext',
    'MaskSamplingMode',
    'SourceKind',
    'SystemExecutionCancelled',
    'SystemExecutionFailed',
    'SystemValidationError',
    'ThinElementLayer',
    'default_backend_registry',

    # Safe canonical and explicitly mapped field I/O
    'AmbiguousMappingError',
    'AmplitudeConvention',
    'ArrayInspection',
    'AxisOrder',
    'DatasetInspection',
    'ExportOptions',
    'ExportReport',
    'FieldFormat',
    'FieldIOError',
    'ImportMapping',
    'ImportMappingError',
    'ImportResult',
    'LengthUnit',
    'MappingConfidence',
    'PhaseUnit',
    'SchemaValidationError',
    'UnsafeDatasetError',
    'UnsupportedFormatError',
    'ZbfBeam',
    'ZbfPilotRays',
    'inspect_field',
    'inspect_zbf',
    'load_field',
    'load_zbf',
    'read_zbf',
    'save_field',
    'save_zbf',
    'write_zbf',

    # Versioned desktop-workbench projects
    'AssetIntegrityError',
    'AssetReference',
    'AssetStatus',
    'ComputeSettings',
    'OptiPropProject',
    'ProjectError',
    'load_project',
    'project_to_domain',
    'save_project',
    'validate_assets',
    
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
    print("- metaatom: Meta-atom database lookup")
    print("- propagation: Propagation algorithms")
    print("- utils: Utility functions")
    print("\nUsage example:")
    print("import optiprop")
    print("field = optiprop.NearField(pixel_size=1e-6)")
    print("lens = optiprop.EqualPathPhase(field)")
    print("prop = optiprop.FresnelPropagation()")
