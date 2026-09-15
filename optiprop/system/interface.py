"""Normal-incidence interfaces between homogeneous optical media."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from ..core import Field2D, Severity, ValidationIssue, ValidationReport
from .layers import FieldGeometry, LayerBase, RunContext, _json_value


class InterfaceModel(str, Enum):
    """Supported dielectric-interface approximations."""

    NORMAL_INCIDENCE_SCALAR = "normal_incidence_scalar"


@dataclass(frozen=True)
class InterfaceLayer(LayerBase):
    """A scalar, normal-incidence interface from ``n1`` to ``n2``.

    OptiProp uses the ``exp(-i omega t)`` convention and treats ``Field2D``
    values as electric-field-like complex amplitudes.  When reflections are
    included, both scalar and Ex/Ey components are multiplied by

    ``t_E = 2 n1 / (n1 + n2)``.

    This is a single-pass model: the reflected field is not returned and no
    multiple-reflection cavity is formed.  A finite slab is represented by an
    interface, a propagation layer in ``n2``, and a second interface.
    """

    n1: complex
    n2: complex
    ignore_reflection: bool = False
    model: InterfaceModel = InterfaceModel.NORMAL_INCIDENCE_SCALAR
    name: str = field(default="Interface", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "n1", _medium_index(self.n1, "n1"))
        object.__setattr__(self, "n2", _medium_index(self.n2, "n2"))
        if not isinstance(self.ignore_reflection, bool):
            raise TypeError("ignore_reflection must be a bool.")
        try:
            model = (
                self.model
                if isinstance(self.model, InterfaceModel)
                else InterfaceModel(self.model)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "model must be 'normal_incidence_scalar'."
            ) from exc
        object.__setattr__(self, "model", model)

    @property
    def transmission_coefficient(self) -> complex:
        """Complex electric-field transmission coefficient."""

        if self.ignore_reflection:
            return 1.0 + 0.0j
        return 2.0 * self.n1 / (self.n1 + self.n2)

    @property
    def reflection_coefficient(self) -> complex:
        """Complex electric-field reflection coefficient."""

        if self.ignore_reflection:
            return 0.0 + 0.0j
        return (self.n1 - self.n2) / (self.n1 + self.n2)

    def parameter_config(self) -> dict[str, Any]:
        return {
            "version": 1,
            "n1": _json_value(self.n1),
            "n2": _json_value(self.n2),
            "ignore_reflection": self.ignore_reflection,
            "model": self.model.value,
        }

    def validate(self, field: Field2D) -> ValidationReport:
        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")

        issues: list[ValidationIssue] = []
        issues.append(
            ValidationIssue(
                severity=Severity.INFO,
                code="interface.normal_incidence_scalar_model",
                message=(
                    "This interface uses one normal-incidence scalar "
                    "coefficient for the entire near field; it does not "
                    "resolve angle-dependent s/p Fresnel coefficients."
                ),
                layer_id=str(self.id),
                parameter_path="model",
            )
        )
        if not cmath.isclose(
            field.medium_index,
            self.n1,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="interface.incident_index_mismatch",
                    message=(
                        "The incoming field medium index does not match the "
                        "interface incident index n1."
                    ),
                    layer_id=str(self.id),
                    parameter_path="n1",
                    suggested_fix=(
                        "Set n1 to the upstream field medium index or insert "
                        "the correct preceding interface."
                    ),
                    details={
                        "field_medium_index": _json_value(field.medium_index),
                        "configured_n1": _json_value(self.n1),
                    },
                )
            )
        if self.ignore_reflection:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="interface.reflection_ignored",
                    message=(
                        "Reflection is ignored; the field amplitude is kept "
                        "unchanged while the medium index is updated."
                    ),
                    layer_id=str(self.id),
                    parameter_path="ignore_reflection",
                )
            )
        elif abs(self.reflection_coefficient) > 1e-12:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="interface.reflected_field_discarded",
                    message=(
                        "The transmitted Fresnel amplitude is applied, but "
                        "the reflected branch and multiple reflections are "
                        "not represented by the forward-only optical system."
                    ),
                    layer_id=str(self.id),
                    parameter_path="ignore_reflection",
                )
            )
        if self.n1.imag < 0.0 or self.n2.imag < 0.0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="interface.active_medium",
                    message=(
                        "Active/gain media are not supported by the interface "
                        "layer; under OptiProp's exp(-i omega t) convention "
                        "the imaginary refractive index must be non-negative."
                    ),
                    layer_id=str(self.id),
                    parameter_path="n1" if self.n1.imag < 0.0 else "n2",
                    suggested_fix=(
                        "Use passive media with non-negative imaginary "
                        "refractive indices."
                    ),
                )
            )
        return ValidationReport(tuple(issues))

    def apply(
        self,
        field: Field2D,
        context: RunContext | None = None,
    ) -> Field2D:
        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        if context is not None and not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext or None.")
        if not self.enabled:
            return field

        validation = self.validate(field)
        validation.raise_for_errors()
        if context is not None and context.cancel_token is not None:
            context.cancel_token.throw_if_cancelled()

        coefficient = torch.as_tensor(
            self.transmission_coefficient,
            dtype=field.dtype,
            device=field.device,
        )
        metadata = dict(field.metadata)
        history = tuple(metadata.get("interface_history", ()))
        metadata["interface_history"] = history + (
            {
                "n1": self.n1,
                "n2": self.n2,
                "transmission_coefficient": self.transmission_coefficient,
                "reflection_ignored": self.ignore_reflection,
            },
        )
        return field.with_data(
            field.data * coefficient,
            medium_index=self.n2,
            metadata=metadata,
        )

    def preview_geometry(self, field: Field2D) -> FieldGeometry:
        """Predict the changed medium without allocating a field plane."""

        if not isinstance(field, Field2D):
            raise TypeError("field must be a Field2D.")
        geometry = FieldGeometry.from_field(field)
        if not self.enabled:
            return geometry
        return FieldGeometry(
            grid=geometry.grid,
            wavelength_m=geometry.wavelength_m,
            medium_index=self.n2,
            components=geometry.components,
            z_m=geometry.z_m,
            dtype=geometry.dtype,
            device=geometry.device,
        )


def _medium_index(value: object, name: str) -> complex:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real or complex number.")
    try:
        normalized = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real or complex number.") from exc
    if not (
        math.isfinite(normalized.real) and math.isfinite(normalized.imag)
    ):
        raise ValueError(f"{name} must be finite.")
    if normalized.real <= 0.0:
        raise ValueError(f"{name}.real must be positive.")
    return normalized


__all__ = ["InterfaceLayer", "InterfaceModel"]
