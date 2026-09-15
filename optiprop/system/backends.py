"""Propagation backend dispatch for ordered optical systems."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..propagation import (
    AngularSpectrumPropagator,
    FresnelPropagator,
    PropagationMethod,
    Propagator,
    RayleighSommerfeldPropagator,
)


@dataclass(frozen=True)
class PropagationBackendRegistry:
    """Immutable mapping from propagation methods to backend instances."""

    backends: Mapping[PropagationMethod, Propagator]

    def __post_init__(self) -> None:
        if not isinstance(self.backends, Mapping):
            raise TypeError("backends must be a mapping.")
        normalized: dict[PropagationMethod, Propagator] = {}
        for method, backend in self.backends.items():
            try:
                method = (
                    method
                    if isinstance(method, PropagationMethod)
                    else PropagationMethod(method)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unknown propagation method {method!r}.") from exc
            if not isinstance(backend, Propagator):
                raise TypeError(
                    f"Backend for {method.value!r} must be a Propagator."
                )
            normalized[method] = backend
        object.__setattr__(self, "backends", MappingProxyType(normalized))

    def resolve(self, method: PropagationMethod | str) -> Propagator:
        """Return the backend registered for ``method``."""

        try:
            normalized = (
                method
                if isinstance(method, PropagationMethod)
                else PropagationMethod(method)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown propagation method {method!r}.") from exc
        try:
            return self.backends[normalized]
        except KeyError as exc:
            raise KeyError(
                f"No propagation backend is registered for {normalized.value!r}."
            ) from exc

    def with_backend(
        self,
        method: PropagationMethod | str,
        backend: Propagator,
    ) -> "PropagationBackendRegistry":
        """Return a registry with one method replaced or added."""

        normalized = (
            method
            if isinstance(method, PropagationMethod)
            else PropagationMethod(method)
        )
        updated = dict(self.backends)
        updated[normalized] = backend
        return PropagationBackendRegistry(updated)


_DEFAULT_BACKEND_REGISTRY: PropagationBackendRegistry | None = None


def default_backend_registry() -> PropagationBackendRegistry:
    """Return the process-wide immutable built-in backend registry."""

    global _DEFAULT_BACKEND_REGISTRY
    if _DEFAULT_BACKEND_REGISTRY is None:
        angular = AngularSpectrumPropagator()
        fresnel = FresnelPropagator()
        rayleigh_sommerfeld = RayleighSommerfeldPropagator()
        _DEFAULT_BACKEND_REGISTRY = PropagationBackendRegistry(
            {
                PropagationMethod.ASM: angular,
                PropagationMethod.BLAS: angular,
                PropagationMethod.FRESNEL_TF: fresnel,
                PropagationMethod.FRESNEL_SCALED: fresnel,
                PropagationMethod.RS_FFT: rayleigh_sommerfeld,
                PropagationMethod.RS_DIRECT: rayleigh_sommerfeld,
            }
        )
    return _DEFAULT_BACKEND_REGISTRY


__all__ = [
    "PropagationBackendRegistry",
    "default_backend_registry",
]
