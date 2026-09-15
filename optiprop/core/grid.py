"""Canonical two-dimensional Cartesian grid definitions.

The numerical core uses image-style array ordering throughout: the last two
tensor dimensions are ``[..., y, x]`` and therefore have shape
``[..., ny, nx]``.  All lengths in this module are expressed in metres.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class Grid2D:
    """A regular, axis-aligned, two-dimensional sampling grid.

    ``x_center`` and ``y_center`` locate the geometric midpoint of the sample
    array.  With this definition, both odd and even grids use the same
    coordinate convention::

        x[i] = (i - (nx - 1) / 2) * dx + x_center
        y[j] = (j - (ny - 1) / 2) * dy + y_center

    Parameters are normalized to built-in ``int`` and ``float`` values during
    construction so that a grid is stable and hashable.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    x_center: float = 0.0
    y_center: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.nx, bool) or not isinstance(self.nx, numbers.Integral):
            raise TypeError("nx must be an integer.")
        if isinstance(self.ny, bool) or not isinstance(self.ny, numbers.Integral):
            raise TypeError("ny must be an integer.")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive.")

        dx = _finite_float(self.dx, "dx")
        dy = _finite_float(self.dy, "dy")
        x_center = _finite_float(self.x_center, "x_center")
        y_center = _finite_float(self.y_center, "y_center")
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("dx and dy must be positive.")

        object.__setattr__(self, "nx", int(self.nx))
        object.__setattr__(self, "ny", int(self.ny))
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        object.__setattr__(self, "x_center", x_center)
        object.__setattr__(self, "y_center", y_center)

    @property
    def shape(self) -> Tuple[int, int]:
        """Canonical spatial tensor shape ``(ny, nx)``."""

        return self.ny, self.nx

    @property
    def sample_span_x(self) -> float:
        """Distance between the first and last X sample centres, in metres."""

        return (self.nx - 1) * self.dx

    @property
    def sample_span_y(self) -> float:
        """Distance between the first and last Y sample centres, in metres."""

        return (self.ny - 1) * self.dy

    @property
    def width(self) -> float:
        """Pixel-edge-to-pixel-edge X extent, in metres."""

        return self.nx * self.dx

    @property
    def height(self) -> float:
        """Pixel-edge-to-pixel-edge Y extent, in metres."""

        return self.ny * self.dy

    def x_coordinates(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the one-dimensional X sample coordinates."""

        _validate_coordinate_dtype(dtype)
        index = torch.arange(self.nx, dtype=dtype, device=device)
        return (index - (self.nx - 1) / 2) * self.dx + self.x_center

    def y_coordinates(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the one-dimensional Y sample coordinates."""

        _validate_coordinate_dtype(dtype)
        index = torch.arange(self.ny, dtype=dtype, device=device)
        return (index - (self.ny - 1) / 2) * self.dy + self.y_center

    # Short aliases keep propagation code readable while the explicit names
    # remain discoverable in public APIs.
    x = x_coordinates
    y = y_coordinates

    def meshgrid(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(X, Y)`` coordinate arrays, each with shape ``(ny, nx)``."""

        x = self.x_coordinates(dtype=dtype, device=device)
        y = self.y_coordinates(dtype=dtype, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        return x_grid, y_grid

    @property
    def sample_extent(self) -> Tuple[float, float, float, float]:
        """Return sample-centre extent ``(xmin, xmax, ymin, ymax)``."""

        half_x = self.sample_span_x / 2
        half_y = self.sample_span_y / 2
        return (
            self.x_center - half_x,
            self.x_center + half_x,
            self.y_center - half_y,
            self.y_center + half_y,
        )

    @property
    def pixel_extent(self) -> Tuple[float, float, float, float]:
        """Return pixel-edge extent ``(xmin, xmax, ymin, ymax)``."""

        half_x = self.width / 2
        half_y = self.height / 2
        return (
            self.x_center - half_x,
            self.x_center + half_x,
            self.y_center - half_y,
            self.y_center + half_y,
        )


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


def _validate_coordinate_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("Coordinate dtype must be torch.float32 or torch.float64.")


__all__ = ["Grid2D"]
