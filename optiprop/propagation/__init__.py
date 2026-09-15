"""Optical propagation APIs.

The original stateful propagation classes remain available unchanged through
this package while OptiProp 2.0 algorithms adopt the immutable contracts in
``base``.
"""

from .base import (
    CancellationToken,
    EvanescentPolicy,
    PaddingMode,
    PaddingSpec,
    PrecisionPolicy,
    PropagationCancelled,
    PropagationMethod,
    PropagationResult,
    PropagationSpec,
    Propagator,
)
from .asm import AngularSpectrumPropagator, propagate_angular_spectrum
from .fresnel import FresnelPropagator, propagate_fresnel
from .rayleigh_sommerfeld import (
    RayleighSommerfeldPropagator,
    propagate_rayleigh_sommerfeld,
)
from .legacy import (
    ASMPropagation,
    FresnelPropagation,
    RayleighSommerfeldPropagation,
)
from .sampling import SamplingReport, assess_sampling
from .moving_window import MovingWindowXZ, scan_moving_window_xz

__all__ = [
    "MovingWindowXZ",
    "scan_moving_window_xz",
    "ASMPropagation",
    "AngularSpectrumPropagator",
    "CancellationToken",
    "EvanescentPolicy",
    "FresnelPropagation",
    "FresnelPropagator",
    "PaddingMode",
    "PaddingSpec",
    "PrecisionPolicy",
    "PropagationCancelled",
    "PropagationMethod",
    "PropagationResult",
    "PropagationSpec",
    "Propagator",
    "RayleighSommerfeldPropagation",
    "RayleighSommerfeldPropagator",
    "SamplingReport",
    "assess_sampling",
    "propagate_angular_spectrum",
    "propagate_fresnel",
    "propagate_rayleigh_sommerfeld",
]
