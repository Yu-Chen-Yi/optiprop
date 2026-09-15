"""Immutable optical-layer contracts and propagation layer."""

from __future__ import annotations

import cmath
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import UUID, uuid4

import torch

from ..core import (
    Field2D,
    Grid2D,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from ..propagation import (
    CancellationToken,
    PrecisionPolicy,
    PropagationResult,
    PropagationSpec,
    Propagator,
)
from ..propagation.sampling import assess_sampling
from .backends import PropagationBackendRegistry, default_backend_registry


@dataclass(frozen=True)
class LayerMetadata:
    """Stable identity and editable display state shared by all layers."""

    id: UUID = field(default_factory=uuid4)
    name: str = "Layer"
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id, "id"))
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")


@dataclass(frozen=True)
class FieldGeometry:
    """A field plane description without allocating an output tensor."""

    grid: Grid2D
    wavelength_m: float
    medium_index: complex
    components: tuple[str, ...]
    z_m: float
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_field(cls, field: Field2D) -> "FieldGeometry":
        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        return cls(
            grid=field.grid,
            wavelength_m=field.wavelength_m,
            medium_index=field.medium_index,
            components=field.components,
            z_m=field.z_m,
            dtype=field.dtype,
            device=field.device,
        )


@dataclass(frozen=True)
class RunContext:
    """Execution services passed to layers without serializing runtime objects."""

    cancel_token: CancellationToken | None = None
    registry: PropagationBackendRegistry = field(
        default_factory=default_backend_registry
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cancel_token is not None and not isinstance(
            self.cancel_token, CancellationToken
        ):
            raise TypeError("cancel_token must be a CancellationToken or None.")
        if not isinstance(self.registry, PropagationBackendRegistry):
            raise TypeError("registry must be a PropagationBackendRegistry.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        if not all(isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings.")
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


@runtime_checkable
class OpticalLayer(Protocol):
    """Structural contract used by :class:`OpticalSystem`."""

    id: UUID
    name: str
    enabled: bool

    def validate(self, field: Field2D) -> ValidationReport:
        ...

    def apply(
        self,
        field: Field2D,
        context: RunContext | None = None,
    ) -> Field2D:
        ...

    def to_config(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True, kw_only=True)
class LayerBase:
    """Frozen base implementing stable UUID/display metadata."""

    id: UUID = field(default_factory=uuid4)
    name: str = "Layer"
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id, "id"))
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")

    @property
    def metadata(self) -> LayerMetadata:
        return LayerMetadata(id=self.id, name=self.name, enabled=self.enabled)

    @property
    def parameter_hash(self) -> str:
        """Digest of physical parameters, excluding UUID and display state."""

        payload = self.parameter_config()
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def parameter_config(self) -> dict[str, Any]:
        raise NotImplementedError

    def validate(self, field: Field2D) -> ValidationReport:
        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        return ValidationReport()

    def apply(
        self,
        field: Field2D,
        context: RunContext | None = None,
    ) -> Field2D:
        raise NotImplementedError

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "id": str(self.id),
            "name": self.name,
            "enabled": self.enabled,
            **self.parameter_config(),
        }


@dataclass(frozen=True)
class PropagationLayer(LayerBase):
    """One homogeneous-medium propagation operation in an optical system."""

    spec: PropagationSpec
    keep_intermediate: bool = True
    name: str = field(default="Propagation", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.spec, PropagationSpec):
            raise TypeError("spec must be a PropagationSpec.")
        if not isinstance(self.keep_intermediate, bool):
            raise TypeError("keep_intermediate must be a bool.")
        # PropagationSpec protects only its top-level mapping.  A layer is a
        # long-lived project object, so recursively snapshot nested option
        # collections at this boundary as well.
        frozen_options = _deep_freeze(self.spec.options)
        object.__setattr__(
            self,
            "spec",
            replace(self.spec, options=frozen_options),
        )
        # Fail at the project-model boundary rather than later during save.
        json.dumps(
            _spec_config(self.spec),
            sort_keys=True,
            allow_nan=False,
        )

    def parameter_config(self) -> dict[str, Any]:
        return {
            "version": 1,
            "spec": _spec_config(self.spec),
        }

    def to_config(self) -> dict[str, Any]:
        config = super().to_config()
        config["keep_intermediate"] = self.keep_intermediate
        return config

    def validate(
        self,
        field: Field2D,
        *,
        backend: Propagator | None = None,
        registry: PropagationBackendRegistry | None = None,
    ) -> ValidationReport:
        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        if backend is not None and registry is not None:
            raise ValueError("Pass either backend or registry, not both.")
        if backend is not None and not isinstance(backend, Propagator):
            raise TypeError("backend must be a Propagator or None.")
        if registry is not None and not isinstance(
            registry, PropagationBackendRegistry
        ):
            raise TypeError("registry must be a PropagationBackendRegistry or None.")
        selected = backend or (registry or default_backend_registry()).resolve(
            self.spec.method
        )
        report = selected.validate(field, self.spec)
        if (
            self.spec.medium_index is not None
            and not cmath.isclose(
                field.medium_index,
                self.spec.medium_index,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            report = report.with_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="medium.interface_required",
                    message=(
                        "The propagation medium differs from the incoming "
                        "field medium. Insert an InterfaceLayer before this "
                        "propagation layer."
                    ),
                    parameter_path="medium_index",
                    suggested_fix=(
                        "Add InterfaceLayer(n1=field.medium_index, "
                        "n2=spec.medium_index) immediately upstream."
                    ),
                    details={
                        "field_medium_index": _json_value(field.medium_index),
                        "propagation_medium_index": _json_value(
                            self.spec.medium_index
                        ),
                    },
                )
            )
        return _address_validation(report, self.id)

    def execute(
        self,
        field: Field2D,
        context: RunContext | None = None,
        *,
        backend: Propagator | None = None,
    ) -> PropagationResult:
        """Execute this layer and return the backend's full result."""

        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        context = RunContext() if context is None else context
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext or None.")
        if backend is not None and not isinstance(backend, Propagator):
            raise TypeError("backend must be a Propagator or None.")
        if not self.enabled:
            return PropagationResult(
                field=field,
                spec=self.spec,
                diagnostics={"skipped": True, "reason": "layer_disabled"},
            )
        selected = backend or context.registry.resolve(self.spec.method)
        validation = self.validate(field, backend=selected)
        validation.raise_for_errors()
        if context.cancel_token is not None:
            context.cancel_token.throw_if_cancelled()
        result = selected.propagate(
            field,
            self.spec,
            cancel_token=context.cancel_token,
        )
        # Backends do not know their owning layer; attach the addressed report.
        addressed = _address_validation(result.validation, self.id)
        if addressed == result.validation:
            return result
        return PropagationResult(
            field=result.field,
            spec=result.spec,
            elapsed_s=result.elapsed_s,
            sampling=result.sampling,
            validation=addressed,
            power_before=result.power_before,
            power_after=result.power_after,
            cache_key=result.cache_key,
            diagnostics=result.diagnostics,
        )

    def apply(
        self,
        field: Field2D,
        context: RunContext | None = None,
    ) -> Field2D:
        if not self.enabled:
            if not isinstance(field, Field2D):
                raise TypeError("field must be a Field2D.")
            return field
        return self.execute(field, context).field

    def preview_geometry(self, field: Field2D) -> FieldGeometry:
        """Predict output plane metadata without running an FFT/integral."""

        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        if not self.enabled:
            return FieldGeometry.from_field(field)
        sampling = assess_sampling(field, self.spec)
        dtype = field.dtype
        if self.spec.precision is PrecisionPolicy.COMPLEX64:
            dtype = torch.complex64
        elif self.spec.precision is PrecisionPolicy.COMPLEX128:
            dtype = torch.complex128
        return FieldGeometry(
            grid=sampling.output_grid,
            wavelength_m=field.wavelength_m,
            medium_index=self.spec.resolved_medium_index(field),
            components=field.components,
            z_m=field.z_m + self.spec.distance_m,
            dtype=dtype,
            device=field.device,
        )


def _address_validation(
    report: ValidationReport,
    layer_id: UUID,
) -> ValidationReport:
    if not isinstance(report, ValidationReport):
        raise TypeError("Backend validate() must return a ValidationReport.")
    addressed = []
    for issue in report:
        path = issue.parameter_path
        if path is not None and not path.startswith("spec."):
            path = f"spec.{path}"
        addressed.append(
            ValidationIssue(
                severity=issue.severity,
                code=issue.code,
                message=issue.message,
                layer_id=str(layer_id),
                parameter_path=path,
                suggested_fix=issue.suggested_fix,
                details=issue.details,
            )
        )
    return ValidationReport(tuple(addressed))


def _spec_config(spec: PropagationSpec) -> dict[str, Any]:
    return {
        "method": spec.method.value,
        "distance_m": spec.distance_m,
        "medium_index": _json_value(spec.medium_index),
        "padding": {
            "mode": spec.padding.mode.value,
            "factor": spec.padding.factor,
            "shape": (
                list(spec.padding.shape) if spec.padding.shape is not None else None
            ),
        },
        "evanescent_policy": spec.evanescent_policy.value,
        "precision": spec.precision.value,
        "output_grid": _json_value(spec.output_grid),
        "options": _json_value(spec.options),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Grid2D):
        return {
            "nx": value.nx,
            "ny": value.ny,
            "dx": value.dx,
            "dy": value.dy,
            "x_center": value.x_center,
            "y_center": value.y_center,
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Serializable mapping keys must be strings.")
        return {
            key: _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False),
        )
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise TypeError(
        f"Value of type {type(value).__name__} is not JSON serializable."
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Propagation option mapping keys must be strings.")
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _uuid(value: UUID | str, name: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be a UUID or UUID string.") from exc


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


__all__ = [
    "FieldGeometry",
    "LayerBase",
    "LayerMetadata",
    "OpticalLayer",
    "PropagationLayer",
    "RunContext",
]
