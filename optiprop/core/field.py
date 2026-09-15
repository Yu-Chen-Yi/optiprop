"""Canonical optical field container for the OptiProp numerical core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import torch

from .grid import Grid2D


SCALAR_COMPONENTS = ("scalar",)
EX_EY_COMPONENTS = ("Ex", "Ey")
_VALID_COMPONENT_SETS = (SCALAR_COMPONENTS, EX_EY_COMPONENTS)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


@dataclass(frozen=True, eq=False)
class Field2D:
    """A logically immutable, regularly sampled coherent optical field.

    The canonical data shape is ``[C, ny, nx]``.  A scalar field has one
    component named ``"scalar"``; a polarized component field has two
    components named ``("Ex", "Ey")``.  A single ``Field2D`` always
    represents one vacuum wavelength and one plane.

    The dataclass and its top-level metadata mapping are immutable.  PyTorch
    tensors cannot be made read-only, so callers must not mutate ``data``
    in-place.  Numerical transforms should return a new field with
    :meth:`with_data`; this preserves autograd graphs without defensive tensor
    copies.
    """

    data: torch.Tensor
    grid: Grid2D
    wavelength_m: float
    medium_index: complex = 1.0
    components: Tuple[str, ...] = SCALAR_COMPONENTS
    z_m: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor.")
        if self.data.dtype not in _COMPLEX_DTYPES:
            raise TypeError(
                "data dtype must be torch.complex64 or torch.complex128; "
                f"got {self.data.dtype}."
            )
        if self.data.ndim != 3:
            raise ValueError(
                "data must have canonical shape [C, ny, nx]; "
                f"got {tuple(self.data.shape)}."
            )
        if not isinstance(self.grid, Grid2D):
            raise TypeError("grid must be a Grid2D.")

        components = tuple(self.components)
        if components not in _VALID_COMPONENT_SETS:
            raise ValueError(
                "components must be exactly ('scalar',) or ('Ex', 'Ey'); "
                f"got {components!r}."
            )
        expected_shape = (len(components), self.grid.ny, self.grid.nx)
        if tuple(self.data.shape) != expected_shape:
            raise ValueError(
                "data shape does not match components and grid: expected "
                f"{expected_shape}, got {tuple(self.data.shape)}."
            )

        wavelength_m = _positive_finite_float(self.wavelength_m, "wavelength_m")
        z_m = _finite_float(self.z_m, "z_m")
        medium_index = _validated_medium_index(self.medium_index)
        metadata = _snapshot_metadata(self.metadata)

        object.__setattr__(self, "components", components)
        object.__setattr__(self, "wavelength_m", wavelength_m)
        object.__setattr__(self, "medium_index", medium_index)
        object.__setattr__(self, "z_m", z_m)
        object.__setattr__(self, "metadata", metadata)

    @property
    def shape(self) -> torch.Size:
        """Full canonical tensor shape ``[C, ny, nx]``."""

        return self.data.shape

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def is_scalar(self) -> bool:
        return self.components == SCALAR_COMPONENTS

    @property
    def is_polarized(self) -> bool:
        return self.components == EX_EY_COMPONENTS

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def real_dtype(self) -> torch.dtype:
        return torch.float32 if self.data.dtype == torch.complex64 else torch.float64

    @property
    def amplitude(self) -> torch.Tensor:
        """Per-component amplitude with shape ``[C, ny, nx]``."""

        return torch.abs(self.data)

    def phase(
        self,
        wrapped: bool = True,
        component: str | int | None = None,
    ) -> torch.Tensor:
        """Return phase in radians.

        Wrapped phase lies in ``[-pi, pi]``.  ``wrapped=False`` performs a
        deterministic sequential unwrap along X and then Y.  This is useful
        for smooth wavefront inspection but is not a branch-cut-aware 2-D
        phase-unwrapping algorithm.

        If ``component`` is omitted the result has shape ``[C, ny, nx]``;
        selecting a component returns ``[ny, nx]``.
        """

        phase = torch.angle(self._select_component(component))
        if wrapped:
            return phase
        return _unwrap_phase_2d(phase)

    def intensity(self, component: str | int | None = None) -> torch.Tensor:
        """Return component or total intensity on the grid.

        With ``component=None``, intensities are summed across components and
        the result has shape ``[ny, nx]``.  Selecting ``"Ex"``, ``"Ey"``,
        ``"scalar"`` or a component index also returns ``[ny, nx]``.
        Intensity is ``|U|**2`` in the field's amplitude convention; no
        impedance or physical-power assumption is applied.
        """

        if component is None:
            return torch.sum(torch.abs(self.data) ** 2, dim=0)
        selected = self._select_component(component)
        return torch.abs(selected) ** 2

    def integrated_intensity(
        self, component: str | int | None = None
    ) -> torch.Tensor:
        """Integrate intensity over area and return a scalar tensor.

        The result remains a tensor on the field's device so autograd and
        asynchronous device workflows are preserved.
        """

        return self.intensity(component=component).sum() * (
            self.grid.dx * self.grid.dy
        )

    def power(self, component: str | int | None = None) -> torch.Tensor:
        """Return the grid-integrated relative intensity.

        Until a field explicitly carries physical electric-field units and an
        impedance convention, this is a relative power-like quantity rather
        than an absolute value in watts.
        """

        return self.integrated_intensity(component=component)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> "Field2D":
        """Return a field with data moved to ``device`` and/or complex dtype."""

        if dtype is not None and dtype not in _COMPLEX_DTYPES:
            raise TypeError(
                "Field2D.to dtype must be torch.complex64 or torch.complex128."
            )
        converted = self.data.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        return self.with_data(converted)

    def with_data(
        self,
        data: torch.Tensor,
        *,
        components: Sequence[str] | None = None,
        grid: Grid2D | None = None,
        wavelength_m: float | None = None,
        medium_index: complex | None = None,
        z_m: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Field2D":
        """Return a new field, replacing data and explicitly supplied context.

        Optional context replacements support propagation and interface layers
        without mutating the input field.  ``None`` means preserve the current
        value; use an empty mapping to clear metadata.
        """

        return Field2D(
            data=data,
            grid=self.grid if grid is None else grid,
            wavelength_m=(
                self.wavelength_m if wavelength_m is None else wavelength_m
            ),
            medium_index=(
                self.medium_index if medium_index is None else medium_index
            ),
            components=(
                self.components if components is None else tuple(components)
            ),
            z_m=self.z_m if z_m is None else z_m,
            metadata=self.metadata if metadata is None else metadata,
        )

    def _select_component(
        self, component: str | int | None
    ) -> torch.Tensor:
        if component is None:
            return self.data
        if isinstance(component, bool):
            raise TypeError("component must be a component name or integer index.")
        if isinstance(component, int):
            if component < 0 or component >= self.component_count:
                raise IndexError(
                    f"component index {component} is out of range for "
                    f"{self.component_count} component(s)."
                )
            return self.data[component]
        if isinstance(component, str):
            try:
                index = self.components.index(component)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown component {component!r}; expected one of "
                    f"{self.components!r}."
                ) from exc
            return self.data[index]
        raise TypeError("component must be a component name or integer index.")


def _unwrap_phase_2d(phase: torch.Tensor) -> torch.Tensor:
    # The final two axes are always Y and X.  Leading component axes, if any,
    # are preserved independently.
    return _unwrap_along(_unwrap_along(phase, dim=-1), dim=-2)


def _unwrap_along(phase: torch.Tensor, dim: int) -> torch.Tensor:
    if phase.shape[dim] < 2:
        return phase

    delta = torch.diff(phase, dim=dim)
    pi = torch.as_tensor(math.pi, dtype=phase.dtype, device=phase.device)
    two_pi = 2 * pi
    wrapped_delta = torch.remainder(delta + pi, two_pi) - pi
    wrapped_delta = torch.where(
        (wrapped_delta == -pi) & (delta > 0), pi, wrapped_delta
    )
    correction = wrapped_delta - delta
    correction = torch.where(torch.abs(delta) < pi, torch.zeros_like(delta), correction)
    correction = torch.cumsum(correction, dim=dim)

    pad_shape = list(phase.shape)
    pad_shape[dim] = 1
    zero = torch.zeros(pad_shape, dtype=phase.dtype, device=phase.device)
    correction = torch.cat((zero, correction), dim=dim)
    return phase + correction


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_finite_float(value: object, name: str) -> float:
    normalized = _finite_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _validated_medium_index(value: object) -> complex:
    if isinstance(value, bool):
        raise TypeError("medium_index must be a real or complex number.")
    try:
        normalized = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("medium_index must be a real or complex number.") from exc
    if not (
        math.isfinite(normalized.real) and math.isfinite(normalized.imag)
    ):
        raise ValueError("medium_index must be finite.")
    if normalized.real <= 0.0:
        raise ValueError("medium_index.real must be positive.")
    return normalized


def _snapshot_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    snapshot = dict(metadata)
    if not all(isinstance(key, str) for key in snapshot):
        raise TypeError("metadata keys must be strings.")
    return MappingProxyType(snapshot)


__all__ = [
    "EX_EY_COMPONENTS",
    "Field2D",
    "SCALAR_COMPONENTS",
]
