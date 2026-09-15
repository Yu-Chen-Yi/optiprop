"""Ordered, immutable optical-system execution."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable
from uuid import UUID, uuid4

import torch

from ..core import (
    Field2D,
    Severity,
    ValidationError,
    ValidationIssue,
    ValidationReport,
)
from ..propagation import CancellationToken, PropagationCancelled
from .backends import PropagationBackendRegistry, default_backend_registry
from .layers import (
    FieldGeometry,
    LayerBase,
    OpticalLayer,
    PropagationLayer,
    RunContext,
)
from .result import (
    LayerExecutionStatus,
    LayerResult,
    OpticalSystemResult,
    SystemExecutionCancelled,
    SystemExecutionFailed,
)


class SystemValidationError(ValidationError):
    """Layer-addressed validation failure with completed partial results."""

    def __init__(
        self,
        report: ValidationReport,
        partial_result: OpticalSystemResult,
    ) -> None:
        if not isinstance(partial_result, OpticalSystemResult):
            raise TypeError("partial_result must be an OpticalSystemResult.")
        self.partial_result = partial_result
        super().__init__(report)


@dataclass(frozen=True)
class OpticalSystem:
    """An immutable ordered sequence of arbitrary optical layers."""

    layers: tuple[OpticalLayer, ...] = ()
    id: UUID = field(default_factory=uuid4, kw_only=True)
    name: str = field(default="Optical system", kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.layers, (str, bytes)):
            raise TypeError("layers must be an iterable of OpticalLayer objects.")
        try:
            layers = tuple(self.layers)
        except TypeError as exc:
            raise TypeError(
                "layers must be an iterable of OpticalLayer objects."
            ) from exc
        for layer in layers:
            if not isinstance(layer, OpticalLayer):
                raise TypeError(
                    "Each layer must implement the OpticalLayer protocol; "
                    f"got {type(layer).__name__}."
                )
            _layer_id(layer)
            _required_text(layer.name, "layer.name")
            if not isinstance(layer.enabled, bool):
                raise TypeError("layer.enabled must be a bool.")
        layer_ids = tuple(_layer_id(layer) for layer in layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Layer IDs must be unique within an optical system.")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "id", _uuid(self.id, "id"))
        object.__setattr__(self, "name", _required_text(self.name, "name"))

    def validate(
        self,
        field: Field2D,
        *,
        registry: PropagationBackendRegistry | None = None,
    ) -> ValidationReport:
        """Collect structural and layer findings without applying any layer."""

        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        registry = registry or default_backend_registry()
        if not isinstance(registry, PropagationBackendRegistry):
            raise TypeError("registry must be a PropagationBackendRegistry or None.")
        return self._validate_layers(field, self.layers, registry)

    def _validate_layers(
        self,
        field: Field2D,
        layers: Iterable[OpticalLayer],
        registry: PropagationBackendRegistry,
    ) -> ValidationReport:
        report = self._structural_validation()
        preview_field = field
        for layer in layers:
            if not layer.enabled:
                continue
            try:
                if isinstance(layer, PropagationLayer):
                    layer_report = layer.validate(
                        preview_field,
                        registry=registry,
                    )
                else:
                    layer_report = _address_generic_validation(
                        layer.validate(preview_field), _layer_id(layer)
                    )
            except KeyError:
                layer_report = ValidationReport(
                    (
                        ValidationIssue(
                            severity=Severity.ERROR,
                            code="system.backend_not_registered",
                            message=(
                                "No numerical backend is registered for this "
                                "propagation method."
                            ),
                            layer_id=str(_layer_id(layer)),
                            parameter_path="spec.method",
                        ),
                    )
                )
            report = report.merge(layer_report)
            if layer_report.is_valid:
                preview_geometry = getattr(layer, "preview_geometry", None)
                if callable(preview_geometry):
                    geometry = preview_geometry(preview_field)
                    if not isinstance(geometry, FieldGeometry):
                        raise TypeError(
                            "OpticalLayer.preview_geometry() must return a "
                            "FieldGeometry."
                        )
                    preview_field = _field_from_geometry(geometry)
        return report

    def execute(
        self,
        input_field: Field2D,
        *,
        context: RunContext | None = None,
        cancel_token: CancellationToken | None = None,
        registry: PropagationBackendRegistry | None = None,
        start_at: UUID | str | None = None,
        stop_after: UUID | str | None = None,
    ) -> OpticalSystemResult:
        """Run an ordered layer range, preserving snapshots and provenance.

        ``start_at`` treats ``input_field`` as the selected layer's input.
        ``stop_after`` includes the selected layer.  Cancellation and failures
        carry a partial result containing only already visited layers.
        """

        if not isinstance(input_field, Field2D):
            raise TypeError("input_field must be a Field2D.")
        context = _resolved_context(context, cancel_token, registry)
        start_index, stop_index = self._execution_range(start_at, stop_after)
        started = time.perf_counter()
        current = input_field
        results: list[LayerResult] = []
        combined_validation = self._structural_validation()

        preflight = self._validate_layers(
            input_field,
            self.layers[start_index:stop_index],
            context.registry,
        )
        if not preflight.is_valid:
            partial = self._result(
                input_field,
                current,
                results,
                preflight,
                started,
            )
            raise SystemValidationError(preflight, partial)

        for layer in self.layers[start_index:stop_index]:
            try:
                if context.cancel_token is not None:
                    context.cancel_token.throw_if_cancelled()
            except PropagationCancelled as exc:
                raise SystemExecutionCancelled(
                    self._result(
                        input_field,
                        current,
                        results,
                        combined_validation,
                        started,
                    ),
                    exc.reason,
                ) from exc

            layer_input = current
            parameter_hash = _layer_parameter_hash(layer)
            if not layer.enabled:
                results.append(
                    LayerResult(
                        layer_id=_layer_id(layer),
                        layer_name=layer.name,
                        layer_type=type(layer).__name__,
                        status=LayerExecutionStatus.SKIPPED,
                        input_field=layer_input,
                        output_field=layer_input,
                        parameter_hash=parameter_hash,
                        diagnostics={"reason": "layer_disabled"},
                    )
                )
                continue

            layer_started = time.perf_counter()
            try:
                if isinstance(layer, PropagationLayer):
                    validation = layer.validate(
                        layer_input, registry=context.registry
                    )
                else:
                    validation = _address_generic_validation(
                        layer.validate(layer_input), _layer_id(layer)
                    )
                combined_validation = combined_validation.merge(validation)
                if not validation.is_valid:
                    partial = self._result(
                        input_field,
                        current,
                        results,
                        combined_validation,
                        started,
                    )
                    raise SystemValidationError(validation, partial)

                if isinstance(layer, PropagationLayer):
                    propagation_result = layer.execute(layer_input, context)
                    current = propagation_result.field
                    elapsed_s = propagation_result.elapsed_s
                    diagnostics = {
                        "method": layer.spec.method.value,
                        "backend": type(
                            context.registry.resolve(layer.spec.method)
                        ).__name__,
                        **dict(propagation_result.diagnostics),
                    }
                    cache_key = propagation_result.cache_key
                else:
                    propagation_result = None
                    current = layer.apply(layer_input, context)
                    if not isinstance(current, Field2D):
                        raise TypeError(
                            "OpticalLayer.apply() must return a Field2D."
                        )
                    elapsed_s = time.perf_counter() - layer_started
                    diagnostics = {}
                    cache_key = None
                results.append(
                    LayerResult(
                        layer_id=_layer_id(layer),
                        layer_name=layer.name,
                        layer_type=type(layer).__name__,
                        status=LayerExecutionStatus.COMPLETED,
                        input_field=layer_input,
                        output_field=current,
                        validation=validation,
                        elapsed_s=elapsed_s,
                        parameter_hash=parameter_hash,
                        diagnostics=diagnostics,
                        propagation_result=propagation_result,
                        cache_key=cache_key,
                    )
                )
                if context.cancel_token is not None:
                    context.cancel_token.throw_if_cancelled()
            except SystemValidationError:
                raise
            except PropagationCancelled as exc:
                raise SystemExecutionCancelled(
                    self._result(
                        input_field,
                        current,
                        results,
                        combined_validation,
                        started,
                    ),
                    exc.reason,
                ) from exc
            except ValidationError as exc:
                addressed = _address_generic_validation(
                    exc.report, _layer_id(layer)
                )
                combined_validation = combined_validation.merge(addressed)
                partial = self._result(
                    input_field,
                    current,
                    results,
                    combined_validation,
                    started,
                )
                raise SystemValidationError(addressed, partial) from exc
            except Exception as exc:
                partial = self._result(
                    input_field,
                    current,
                    results,
                    combined_validation,
                    started,
                )
                raise SystemExecutionFailed(
                    _layer_id(layer), partial, exc
                ) from exc

        return self._result(
            input_field,
            current,
            results,
            combined_validation,
            started,
        )

    run = execute

    def get(self, layer_id: UUID | str) -> OpticalLayer:
        return self.layers[self.index(layer_id)]

    def index(self, layer_id: UUID | str) -> int:
        normalized = _uuid(layer_id, "layer_id")
        for index, layer in enumerate(self.layers):
            if _layer_id(layer) == normalized:
                return index
        raise KeyError(f"No layer exists with id {normalized}.")

    def append(self, layer: OpticalLayer) -> "OpticalSystem":
        return replace(self, layers=self.layers + (_validated_layer(layer),))

    def insert(self, index: int, layer: OpticalLayer) -> "OpticalSystem":
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if index < 0 or index > len(self.layers):
            raise IndexError("insert index is out of range.")
        updated = self.layers[:index] + (_validated_layer(layer),) + self.layers[index:]
        return replace(self, layers=updated)

    def remove(self, layer_id: UUID | str) -> "OpticalSystem":
        index = self.index(layer_id)
        return replace(self, layers=self.layers[:index] + self.layers[index + 1 :])

    def replace(
        self,
        layer_id: UUID | str,
        replacement: OpticalLayer,
        *,
        preserve_id: bool = True,
    ) -> "OpticalSystem":
        index = self.index(layer_id)
        replacement = _validated_layer(replacement)
        if preserve_id and _layer_id(replacement) != _layer_id(self.layers[index]):
            if not isinstance(replacement, LayerBase):
                raise TypeError(
                    "preserve_id requires a dataclass derived from LayerBase."
                )
            replacement = replace(replacement, id=_layer_id(self.layers[index]))
        updated = list(self.layers)
        updated[index] = replacement
        return replace(self, layers=tuple(updated))

    def move(self, layer_id: UUID | str, new_index: int) -> "OpticalSystem":
        if isinstance(new_index, bool) or not isinstance(new_index, int):
            raise TypeError("new_index must be an integer.")
        if new_index < 0 or new_index >= len(self.layers):
            raise IndexError("new_index is out of range.")
        old_index = self.index(layer_id)
        updated = list(self.layers)
        layer = updated.pop(old_index)
        updated.insert(new_index, layer)
        return replace(self, layers=tuple(updated))

    def update(self, layer_id: UUID | str, **changes: Any) -> "OpticalSystem":
        layer = self.get(layer_id)
        if not isinstance(layer, LayerBase):
            raise TypeError("update requires a dataclass derived from LayerBase.")
        if "id" in changes:
            raise ValueError("update cannot change a layer's stable id.")
        return self.replace(layer_id, replace(layer, **changes))

    def set_enabled(
        self, layer_id: UUID | str, enabled: bool
    ) -> "OpticalSystem":
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool.")
        return self.update(layer_id, enabled=enabled)

    def duplicate(
        self,
        layer_id: UUID | str,
        *,
        name: str | None = None,
    ) -> "OpticalSystem":
        index = self.index(layer_id)
        layer = self.layers[index]
        if not isinstance(layer, LayerBase):
            raise TypeError("duplicate requires a dataclass derived from LayerBase.")
        duplicate_name = (
            _required_text(name, "name") if name is not None else f"{layer.name} copy"
        )
        duplicate = replace(layer, id=uuid4(), name=duplicate_name)
        return self.insert(index + 1, duplicate)

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "OpticalSystem",
            "version": 1,
            "id": str(self.id),
            "name": self.name,
            "layers": [layer.to_config() for layer in self.layers],
        }

    def _execution_range(
        self,
        start_at: UUID | str | None,
        stop_after: UUID | str | None,
    ) -> tuple[int, int]:
        start = 0 if start_at is None else self.index(start_at)
        stop = len(self.layers) if stop_after is None else self.index(stop_after) + 1
        if stop < start:
            raise ValueError("stop_after must not precede start_at.")
        return start, stop

    def _structural_validation(self) -> ValidationReport:
        seen: set[UUID] = set()
        issues = []
        for layer in self.layers:
            layer_id = _layer_id(layer)
            if layer_id in seen:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="system.duplicate_layer_id",
                        message="Layer IDs must be unique within an optical system.",
                        layer_id=str(layer_id),
                        parameter_path="id",
                        suggested_fix="Assign a new UUID to the duplicated layer.",
                    )
                )
            seen.add(layer_id)
        return ValidationReport(tuple(issues))

    def _result(
        self,
        input_field: Field2D,
        final_field: Field2D,
        results: Iterable[LayerResult],
        validation: ValidationReport,
        started: float,
    ) -> OpticalSystemResult:
        return OpticalSystemResult(
            system_id=self.id,
            input_field=input_field,
            final_field=final_field,
            layer_results=tuple(results),
            validation=validation,
            elapsed_s=time.perf_counter() - started,
        )


def _resolved_context(
    context: RunContext | None,
    cancel_token: CancellationToken | None,
    registry: PropagationBackendRegistry | None,
) -> RunContext:
    if context is not None:
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext or None.")
        if cancel_token is not None or registry is not None:
            raise ValueError(
                "cancel_token/registry cannot be combined with an explicit context."
            )
        return context
    if cancel_token is not None and not isinstance(cancel_token, CancellationToken):
        raise TypeError("cancel_token must be a CancellationToken or None.")
    if registry is not None and not isinstance(
        registry, PropagationBackendRegistry
    ):
        raise TypeError("registry must be a PropagationBackendRegistry or None.")
    return RunContext(
        cancel_token=cancel_token,
        registry=registry or default_backend_registry(),
    )


def _field_from_geometry(geometry: FieldGeometry) -> Field2D:
    """Create a zero-stride validation field without allocating a full plane."""

    if not isinstance(geometry, FieldGeometry):
        raise TypeError("geometry must be a FieldGeometry.")
    scalar = torch.zeros(
        (),
        dtype=geometry.dtype,
        device=geometry.device,
    )
    data = scalar.expand(
        len(geometry.components),
        geometry.grid.ny,
        geometry.grid.nx,
    )
    return Field2D(
        data=data,
        grid=geometry.grid,
        wavelength_m=geometry.wavelength_m,
        medium_index=geometry.medium_index,
        components=geometry.components,
        z_m=geometry.z_m,
        metadata={"_optiprop_geometry_preview": True},
    )


def _address_generic_validation(
    report: ValidationReport,
    layer_id: UUID,
) -> ValidationReport:
    if not isinstance(report, ValidationReport):
        raise TypeError("OpticalLayer.validate() must return a ValidationReport.")
    return ValidationReport(
        tuple(
            ValidationIssue(
                severity=issue.severity,
                code=issue.code,
                message=issue.message,
                layer_id=str(layer_id),
                parameter_path=issue.parameter_path,
                suggested_fix=issue.suggested_fix,
                details=issue.details,
            )
            for issue in report
        )
    )


def _layer_parameter_hash(layer: OpticalLayer) -> str:
    value = getattr(layer, "parameter_hash", None)
    if isinstance(value, str) and value:
        return value
    config = dict(layer.to_config())
    for key in ("id", "name", "enabled"):
        config.pop(key, None)
    try:
        encoded = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{type(layer).__name__}.to_config() must return JSON-native data."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _validated_layer(layer: OpticalLayer) -> OpticalLayer:
    if not isinstance(layer, OpticalLayer):
        raise TypeError("layer must implement the OpticalLayer protocol.")
    return layer


def _layer_id(layer: OpticalLayer) -> UUID:
    return _uuid(layer.id, "layer.id")


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
    "OpticalSystem",
    "SystemValidationError",
]
