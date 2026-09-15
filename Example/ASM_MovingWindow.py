"""Run from the clean project: python Example/ASM_MovingWindow.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
from optiprop.core import Field2D, Grid2D
from optiprop.propagation import (PropagationSpec, PaddingSpec,
                                 scan_moving_window_xz)

out = Path(__file__).resolve().parents[1] / 'output' / 'asm_moving_window'
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
grid = Grid2D(129, 97, 0.5e-6, 0.5e-6)
x, y = grid.meshgrid()
angle = np.deg2rad(10)
u = torch.exp(-(x*x+y*y)/(8e-6)**2) * torch.exp(1j*2*np.pi/1.31e-6*np.sin(angle)*x)
field = Field2D(u[None], grid, 1.31e-6, z_m=2e-6)
spec = PropagationSpec('asm', 0, padding=PaddingSpec.by_factor(4))
distances = np.linspace(0, 180e-6, 61)
fixed = scan_moving_window_xz(field,spec,distances)
moving = scan_moving_window_xz(field,spec,distances,slope_x=np.tan(angle))
plt.rcParams.update({'font.size':16,'axes.titlesize':18,'axes.labelsize':17})
fig, axes = plt.subplots(1,3,figsize=(19,7))
for ax, result, local, title in zip(axes, [fixed,moving,moving],
        [False,False,True], ['Fixed window','Moving window: global','Moving window: local']):
    xx = result.x_m.numpy()*1e6
    if local:
        xx = np.broadcast_to(grid.x().numpy()*1e6,xx.shape)
    im=ax.pcolormesh(xx,result.z_m.numpy()*1e6,
                    result.field[0].abs().square().numpy(),shading='auto',cmap='turbo')
    ax.set(title=title,xlabel='Local ξ (µm)' if local else 'Global x (µm)',ylabel='Global z (µm)')
    cax=make_axes_locatable(ax).append_axes('right',size='5%',pad=0.12)
    fig.colorbar(im,cax=cax,label='Intensity (a.u.)')
fig.tight_layout()
fig.savefig(out/'comparison.png',dpi=200)
np.savez(out/'moving_xz.npz',field=moving.field.numpy(),x_m=moving.x_m.numpy(),
         z_m=moving.z_m.numpy(),y_m=moving.y_m)
print(out)
