"""Parallel moving-window ASM. All lengths are metres, slopes dimensionless.

This is not tilted-plane propagation. Padding/convergence must be checked;
FFT translation does not eliminate periodic replicas or spectral aliasing.
"""
from dataclasses import dataclass, replace
import math
import torch
from .asm import propagate_angular_spectrum


@dataclass(frozen=True)
class MovingWindowXZ:
    field: torch.Tensor  # [component, z, x]
    x_m: torch.Tensor  # [z, x], global coordinates
    z_m: torch.Tensor  # [z, x], global coordinates
    y_m: float


def scan_moving_window_xz(field, spec, distances_m, *, slope_x=0.0,
                          offset_x_m=0.0, y_index=None):
    """Sample an input-grid row at each z; return the real global coordinates.

    Even grids have no centre sample at y=0: the selected row coordinate is
    returned explicitly. Uses full 2D IFFT for parity-independent correctness.
    Every distance is measured from the original input, not the previous row.
    """
    if not math.isfinite(slope_x) or not math.isfinite(offset_x_m):
        raise ValueError('Slope and offset must be finite.')
    distances = [float(z) for z in distances_m]
    if not distances or not all(math.isfinite(z) for z in distances):
        raise ValueError('Provide nonempty finite distances.')
    iy = field.grid.ny // 2 if y_index is None else y_index
    if not isinstance(iy, int) or not 0 <= iy < field.grid.ny:
        raise ValueError('y_index must select an input grid row.')
    rows, xs, zs = [], [], []
    for distance in distances:
        result = propagate_angular_spectrum(
            field, replace(spec, distance_m=distance),
            window_shift_m=(offset_x_m + slope_x * distance, 0.0))
        rows.append(result.field.data[:, iy, :])
        x = result.field.grid.x(device=field.data.device)
        xs.append(x)
        zs.append(torch.full_like(x, field.z_m + distance))
    return MovingWindowXZ(torch.stack(rows, dim=1), torch.stack(xs),
                          torch.stack(zs), float(field.grid.y()[iy]))
