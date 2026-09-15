import math
import pytest
import torch
from optiprop.core import Field2D, Grid2D
from optiprop.propagation import (PropagationSpec, PaddingSpec,
    propagate_angular_spectrum, scan_moving_window_xz)


@pytest.mark.parametrize('shape', [(17, 24), (18, 23)])
def test_fractional_shift_plane_wave(shape):
    ny, nx = shape
    grid = Grid2D(nx, ny, 1e-6, 1.2e-6, x_center=7e-6)
    x, y = grid.meshgrid()
    kx, ky = 2*math.pi/(nx*grid.dx), 2*math.pi/(ny*grid.dy)
    u = torch.exp(1j*(kx*x+ky*y))[None]
    field = Field2D(u, grid, 0.5e-6)
    spec = PropagationSpec('asm', 2e-6, padding=PaddingSpec.none())
    a, b = 0.37e-6, -0.29e-6
    result = propagate_angular_spectrum(field, spec, window_shift_m=(a,b))
    kz = math.sqrt((2*math.pi/field.wavelength_m)**2-kx*kx-ky*ky)
    torch.testing.assert_close(result.field.data, u*complex(
        math.cos(kz*spec.distance_m+kx*a+ky*b),
        math.sin(kz*spec.distance_m+kx*a+ky*b)))
    assert result.field.grid.x_center == grid.x_center+a
    scan = scan_moving_window_xz(field, spec, [0, 2e-6], slope_x=0.1)
    torch.testing.assert_close(scan.x_m[1]-scan.x_m[0],
                               torch.full((nx,), 0.2e-6, dtype=torch.float64))


def test_zero_distance_integer_shift_and_autograd():
    grid = Grid2D(16, 12, 1e-6, 1e-6)
    u = torch.randn(2,12,16,dtype=torch.complex128,requires_grad=True)
    field = Field2D(u,grid,0.5e-6,components=('Ex','Ey'))
    spec = PropagationSpec('asm',0,padding=PaddingSpec.none())
    out = propagate_angular_spectrum(field,spec,window_shift_m=(grid.dx,0))
    torch.testing.assert_close(out.field.data,torch.roll(u,-1,-1))
    out.field.data.abs().square().sum().backward()
    assert torch.isfinite(u.grad).all()
    with pytest.raises(ValueError):
        propagate_angular_spectrum(field, PropagationSpec('blas',1e-6),
                                   window_shift_m=(1e-6,0))
