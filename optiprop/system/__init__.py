"""Immutable multilayer optical-system APIs.

Execution is deliberately field-in/field-out: a source or importer creates the
initial :class:`~optiprop.core.Field2D`, then enabled layers consume outputs in
list order.  Disabled layers are transparent but still produce a ``SKIPPED``
snapshot so UI row state and result provenance remain aligned.

Validation runs immediately before each layer, using the actual upstream
field.  ``start_at`` therefore assumes the supplied input is already the field
at that layer's entrance.  This package retains complete field snapshots;
content-addressed caching and field-summary eviction are deferred to the
simulation-service work package rather than hidden in the domain model.
"""

from .backends import PropagationBackendRegistry, default_backend_registry
from .layers import (
    FieldGeometry,
    LayerBase,
    LayerMetadata,
    OpticalLayer,
    PropagationLayer,
    RunContext,
)
from .interface import InterfaceLayer, InterfaceModel
from .optical_system import OpticalSystem, SystemValidationError
from .sources import IncidentSource, SourceKind
from .thin_elements import (
    ApertureLayer,
    ApertureShape,
    ApertureSpec,
    ComplexMaskLayer,
    IdealLensLayer,
    LensPhaseModel,
    MaskSamplingMode,
    ThinElementLayer,
)
from .result import (
    LayerExecutionStatus,
    LayerResult,
    OpticalSystemResult,
    SystemExecutionCancelled,
    SystemExecutionFailed,
)

__all__ = [
    "FieldGeometry",
    "ApertureLayer",
    "ApertureShape",
    "ApertureSpec",
    "ComplexMaskLayer",
    "IdealLensLayer",
    "InterfaceLayer",
    "InterfaceModel",
    "LensPhaseModel",
    "IncidentSource",
    "LayerBase",
    "LayerExecutionStatus",
    "LayerMetadata",
    "LayerResult",
    "OpticalLayer",
    "OpticalSystem",
    "OpticalSystemResult",
    "PropagationBackendRegistry",
    "PropagationLayer",
    "RunContext",
    "MaskSamplingMode",
    "SourceKind",
    "SystemExecutionCancelled",
    "SystemExecutionFailed",
    "SystemValidationError",
    "ThinElementLayer",
    "default_backend_registry",
]
