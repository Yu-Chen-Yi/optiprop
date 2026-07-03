"""
AxiCubicPhase 驗證腳本 (只跑 section 5 + section 6)
====================================================

讀取先前優化存下的 axicubic_opt_result.json (axi / alpha 等參數),
不重新優化, 直接用最佳相位遮罩做驗證:

    開頭     : 畫出最佳相位遮罩的振幅 / 相位分布並存成圖片
    Section 5 : 掃描 N 個波長 linspace(min_lam, max_lam, N_lam) 的 xy 聚焦強度 + FWHM
    Section 6 : 掃描 N 個波長 linspace(min_lam, max_lam, N_lam) 的 x–z 縱向強度

用法:
    python val.py
可在下方 CONFIG 區塊調整讀取的 json、視窗大小、掃描波長與 z 掃描範圍。
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # 無視窗環境, 直接輸出圖檔
import matplotlib.pyplot as plt

import optiprop
from optiprop import NearField, AxiCubicPhase, ASMPropagation

# ======================================================================
# CONFIG (可自行調整)
# ======================================================================
RESULT_JSON      = 'result5/axicubic_opt_result.json'   # 要讀取的優化結果
FIELD_PAD_FACTOR = 2.0    # 模擬視窗 = lens_D * 此倍率 (需與優化時一致)
N_MEDIUM         = 1.0    # 折射率

# --- 掃描波長設定 (section 5 與 6 共用): linspace(MIN_LAM, MAX_LAM, N_LAM) ---
MIN_LAM = 8e-6            # 掃描最短波長 (m)
MAX_LAM = 12e-6           # 掃描最長波長 (m)
N_LAM   = 7              # 掃描波長數量 (任意正整數)

# --- Section 5: xy 聚焦圖的顯示範圍 (µm) ---
XY_VIEW_UM = 150

# --- Section 6: x–z 掃描設定 ---
Z_SCAN_MIN = 10e-3       # z 掃描起點 (m)
Z_SCAN_MAX = 20e-3       # z 掃描終點 (m)
Z_SCAN_N   = 120         # z 掃描點數
XZ_VIEW_UM = 200         # x–z 圖的 x 顯示範圍 (µm)

# --- 開頭遮罩振幅 / 相位分布圖 ---
SAVE_FIELD_FIG = True    # 是否畫出並存出近場遮罩的振幅 / 相位分布圖

# ======================================================================
# 0. 環境 + 讀取優化結果
# ======================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
cdtype = torch.complex64

with open(RESULT_JSON, 'r', encoding='utf-8') as f:
    res = json.load(f)

axi_opt   = float(res['axi'])
alpha_opt = float(res['alpha'])
pixel_size = float(res['pixel_size_um']) * 1e-6
lens_D     = float(res['lens_D_mm']) * 1e-3
z_focus    = float(res['z_focus_mm']) * 1e-3
wavelengths = [w * 1e-6 for w in res['wavelengths_um']]   # 設計波長 (section 5 用)
field_size  = lens_D * FIELD_PAD_FACTOR

OUTDIR = os.path.dirname(os.path.abspath(RESULT_JSON))

print(f'optiprop {optiprop.__version__} | torch {torch.__version__} | device = {device}')
print(f'loaded  {RESULT_JSON}')
print(f'  axi = {axi_opt:.3f}, alpha = {alpha_opt:.3f}')
print(f'  design wavelengths = {[f"{w*1e6:.2f}um" for w in wavelengths]}')
print(f'  z_focus = {z_focus*1e3:.2f} mm, lens_D = {lens_D*1e3:.3f} mm, pixel = {pixel_size*1e6:.2f} um')

# ======================================================================
# 1. 網格 + 重建最佳相位遮罩
# ======================================================================
nf = NearField(pixel_size=pixel_size, field_Lx=field_size, field_Ly=field_size,
               dtype=dtype, device=device)
X, Y = nf.X, nf.Y
Nx, Ny = nf.Nx, nf.Ny
print(f'Grid: {Nx} x {Ny} pixels, field = {field_size*1e3:.2f} mm')

elem = AxiCubicPhase(nf)
U0_opt = elem.calculate_phase(
    axi=torch.tensor(axi_opt, device=device),
    alpha=torch.tensor(alpha_opt, device=device),
    design_lambda=wavelengths[len(wavelengths) // 2], lens_diameter=lens_D,
    lens_center=[0, 0], aperture_type='circle',
    aperture_size=[lens_D, lens_D],
).detach()

x_um = X[Ny // 2, :].detach().cpu().numpy() * 1e6

# 兩個 section 共用的掃描波長
scan_wavelengths = np.linspace(MIN_LAM, MAX_LAM, N_LAM)

# ======================================================================
# 2. 開頭: 畫出遮罩的振幅 / 相位分布並存成圖片
# ======================================================================
if SAVE_FIELD_FIG:
    amp = U0_opt.abs().detach().cpu().numpy()
    pha = torch.angle(U0_opt).detach().cpu().numpy()   # rad, -pi~pi
    ext = [x_um.min(), x_um.max(), x_um.min(), x_um.max()]

    lens_R_um = lens_D / 2 * 1e6   # 只顯示透鏡孔徑範圍, 不畫外圍 padding

    fig, axf = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axf[0].imshow(amp, extent=ext, cmap='gray', origin='lower')
    axf[0].set_title(f'Amplitude  (axi={axi_opt:.2f}, alpha={alpha_opt:.2f})')
    axf[0].set_xlabel('x (µm)'); axf[0].set_ylabel('y (µm)')
    axf[0].set_xlim(-lens_R_um, lens_R_um); axf[0].set_ylim(-lens_R_um, lens_R_um)
    plt.colorbar(im0, ax=axf[0], fraction=0.046)

    im1 = axf[1].imshow(pha, extent=ext, cmap='twilight', origin='lower',
                        vmin=-np.pi, vmax=np.pi)
    axf[1].set_title('Phase (rad)')
    axf[1].set_xlabel('x (µm)'); axf[1].set_ylabel('y (µm)')
    axf[1].set_xlim(-lens_R_um, lens_R_um); axf[1].set_ylim(-lens_R_um, lens_R_um)
    plt.colorbar(im1, ax=axf[1], fraction=0.046)

    plt.tight_layout()
    p = os.path.join(OUTDIR, 'axicubic_val_field.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p)

# ======================================================================
# 5. 驗證: 掃描 N 個波長在 z = z_focus 的 xy 聚焦能量分布 + FWHM
# ======================================================================
ncols = min(N_LAM, 4)
nrows = int(np.ceil(N_LAM / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
axes_flat = axes.ravel()

for j, lam in enumerate(scan_wavelengths):
    prop = ASMPropagation(propagation_wavelength=float(lam), propagation_distance=z_focus,
                          n=N_MEDIUM, dtype=dtype, device=device)
    prop.set_input_field(U0_opt, pixel_size)
    prop.propagate()
    I = prop.get_intensity.detach().cpu().numpy()

    # 由中心線估算 FWHM
    line = I[Ny // 2, :]
    line_n = line / line.max()
    half = line_n >= 0.5
    fwhm_txt = f'FWHM ≈ {x_um[half].max() - x_um[half].min():.1f} µm' if half.any() else ''

    axm = axes_flat[j]
    im = axm.imshow(I, extent=[x_um.min(), x_um.max(), x_um.min(), x_um.max()],
                    cmap='turbo', origin='lower')
    axm.set_title(f'{lam*1e6:.2f} µm  @ z={z_focus*1e3:.2f}mm\n{fwhm_txt}')
    axm.set_xlabel('x (µm)'); axm.set_ylabel('y (µm)')
    axm.set_xlim(-XY_VIEW_UM, XY_VIEW_UM); axm.set_ylim(-XY_VIEW_UM, XY_VIEW_UM)
    plt.colorbar(im, ax=axm, fraction=0.046)

for k in range(N_LAM, len(axes_flat)):
    axes_flat[k].axis('off')

plt.tight_layout()
p = os.path.join(OUTDIR, 'axicubic_val_focus_xy.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {p}  (swept {N_LAM} wavelengths: {MIN_LAM*1e6:.2f}–{MAX_LAM*1e6:.2f} µm)')

# ======================================================================
# 6. x–z 縱向強度: 掃描 N 個波長 linspace(MIN_LAM, MAX_LAM, N_LAM)
# ======================================================================
z_range = np.linspace(Z_SCAN_MIN, Z_SCAN_MAX, Z_SCAN_N)

ncols = min(N_LAM, 4)
nrows = int(np.ceil(N_LAM / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
axes_flat = axes.ravel()

for j, lam in enumerate(scan_wavelengths):
    prop = ASMPropagation(propagation_wavelength=float(lam), propagation_distance=z_focus,
                          n=N_MEDIUM, dtype=dtype, device=device)
    prop.set_input_field(U0_opt, pixel_size)
    prop.propagate_xz(z_range=z_range)
    Ixz = (prop.get_output_UZ.abs() ** 2).detach().cpu().numpy().T   # [Nx, Nz]
    axm = axes_flat[j]
    im = axm.imshow(Ixz, extent=[z_range.min()*1e3, z_range.max()*1e3, x_um.min(), x_um.max()],
                    cmap='turbo', origin='lower', aspect='auto')
    axm.axvline(z_focus*1e3, color='w', ls='--', lw=1)
    axm.set_ylim(-XZ_VIEW_UM, XZ_VIEW_UM)
    axm.set_title(f'{lam*1e6:.2f} µm  (x–z)'); axm.set_xlabel('z (mm)'); axm.set_ylabel('x (µm)')
    plt.colorbar(im, ax=axm, fraction=0.046)

for k in range(N_LAM, len(axes_flat)):
    axes_flat[k].axis('off')

plt.tight_layout()
p = os.path.join(OUTDIR, 'axicubic_val_focus_xz.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {p}  (swept {N_LAM} wavelengths: {MIN_LAM*1e6:.2f}–{MAX_LAM*1e6:.2f} µm)')
