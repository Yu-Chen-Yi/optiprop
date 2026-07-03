"""
AxiCubicPhase 多波長聚焦優化 (Autograd + Adam)
=================================================

使用 PyTorch autograd + Adam 對 optiprop.AxiCubicPhase 的兩個參數
    axi   : 軸錐 (axicon) 項
    alpha : 立方 (cubic) 項
做梯度優化。

相位模型 (與波長無關, 單一相位遮罩):
    phi(x,y) = axi * r + alpha * (|x_hat^3| + |y_hat^3|)
    x_hat = x/(D/2), y_hat = y/(D/2), r = sqrt(x_hat^2 + y_hat^2)

優化目標:
    同一個相位分布在波長 8 µm、9.6 µm、12 µm 都能在 z = 2.4 mm 處聚焦。

規格:
    圓形透鏡孔徑 D      = 2 mm
    取樣週期 pixel_size = 5 µm
    聚焦距離 z          = 2.4 mm
    波長                = 8, 9.6, 12 µm

軸錐項會產生類 Bessel 的延焦深 (EDOF) 焦線, 是達成多波長/消色差聚焦的關鍵;
立方項提供額外自由度。優化器會自動在三個波長間取得最佳折衷。
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

# ----------------------------------------------------------------------
# 0. 環境
# ----------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
cdtype = torch.complex64
torch.manual_seed(0)

OUTDIR = 'result'
os.makedirs(OUTDIR, exist_ok=True)
print(f'optiprop {optiprop.__version__} | torch {torch.__version__} | device = {device}')

# ----------------------------------------------------------------------
# 1. 基本參數與網格設定
# ----------------------------------------------------------------------
pixel_size  = 2e-6        # 取樣週期 5 µm
lens_D      = 2.286e-3 * 2        # 圓形透鏡孔徑 2 mm
field_size  = 2.286e-3 * 2 * 2      # 模擬視窗 4 mm (孔徑外留 padding 給 ASM)
z_focus     = 16.76e-3      # 目標聚焦距離 2.4 mm
wavelengths = [8e-6, 9.6e-6, 12e-6]   # 三個設計波長
n_medium    = 1.0
sigma_focus = 50e-6       # 焦點權重高斯半寬 (量化中心聚焦能量)

nf = NearField(pixel_size=pixel_size, field_Lx=field_size, field_Ly=field_size,
               dtype=dtype, device=device)
X, Y = nf.X, nf.Y
Nx, Ny = nf.Nx, nf.Ny
print(f'Grid: {Nx} x {Ny} pixels, field = {field_size*1e3:.2f} mm, pixel = {pixel_size*1e6:.1f} µm')

# 用來建立相位遮罩的元件 (每次迭代以 tensor 參數呼叫 calculate_phase)
elem = AxiCubicPhase(nf)

# 中心聚焦能量的高斯權重 W(x,y)
W = torch.exp(-(X**2 + Y**2) / (2 * sigma_focus**2))


def make_transfer(lam, z):
    """預先計算某波長在距離 z 的 ASM 傳遞函數 H (與 axi/alpha 無關)。"""
    wvl = lam / n_medium
    k = 2 * torch.pi / wvl
    fx = torch.fft.fftfreq(Nx, d=pixel_size, dtype=dtype, device=device)
    fy = torch.fft.fftfreq(Ny, d=pixel_size, dtype=dtype, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    sqrt_term = torch.sqrt(1 - (wvl**2) * (FX**2 + FY**2) + 0j)
    H = torch.exp(1j * k * z * sqrt_term)
    return H.to(cdtype)


H_list = [make_transfer(lam, z_focus) for lam in wavelengths]
print('Transfer functions ready for', [f'{l*1e6:.1f}µm' for l in wavelengths])


# ----------------------------------------------------------------------
# 2. 前向模型與損失函數
# ----------------------------------------------------------------------
def build_field(axi, alpha):
    """以可微分的 tensor 參數建立複數場 U0 (含圓形孔徑)。"""
    return elem.calculate_phase(
        axi=axi, alpha=alpha,
        design_lambda=wavelengths[1], lens_diameter=lens_D,
        lens_center=[0, 0], aperture_type='circle',
        aperture_size=[lens_D, lens_D],
    )


def propagate(U0, H):
    return torch.fft.ifft2(torch.fft.fft2(U0) * H)


def focal_concentration(axi, alpha):
    """回傳 (loss, 每個波長的中心能量比例 list)。"""
    U0 = build_field(axi, alpha)
    concs = []
    for H in H_list:
        Uz = propagate(U0, H)
        I = (Uz.abs()) ** 2
        conc = (I * W).sum() / (I.sum() + 1e-12)   # 中心聚焦能量比例 (0~1)
        concs.append(conc)
    # 幾何平均 (log 平均): 重罰任何一個被犧牲的波長, 強制三個波長都要聚焦
    loss = -torch.stack([torch.log(c + 1e-6) for c in concs]).mean()
    return loss, concs


# 快速自檢: 初始值可微分且有梯度
_axi = torch.tensor(-220.0, device=device, requires_grad=True)
_alpha = torch.tensor(0.0, device=device, requires_grad=True)
_loss, _ = focal_concentration(_axi, _alpha)
_loss.backward()
print(f'sanity check  loss={_loss.item():.4f}  '
      f'grad_axi={_axi.grad.item():.3e}  grad_alpha={_alpha.grad.item():.3e}')


# ----------------------------------------------------------------------
# 3. Adam 梯度優化
# ----------------------------------------------------------------------
axi   = torch.tensor(-220.0, device=device, dtype=dtype, requires_grad=True)
alpha = torch.tensor(   0.0, device=device, dtype=dtype, requires_grad=True)

n_iter = 400
optimizer = torch.optim.Adam([axi, alpha], lr=1.5)

hist = {'loss': [], 'axi': [], 'alpha': [], 'conc': []}
for it in range(n_iter):
    optimizer.zero_grad()
    loss, concs = focal_concentration(axi, alpha)
    loss.backward()
    optimizer.step()

    hist['loss'].append(loss.item())
    hist['axi'].append(axi.item())
    hist['alpha'].append(alpha.item())
    hist['conc'].append([c.item() for c in concs])
    if it % 20 == 0 or it == n_iter - 1:
        cs = ' '.join(f'{c.item():.3f}' for c in concs)
        print(f'iter {it:3d} | loss {loss.item():+.4f} | axi {axi.item():8.2f} | '
              f'alpha {alpha.item():8.2f} | conc[8,9.6,12um]= {cs}')

axi_opt, alpha_opt = axi.item(), alpha.item()
print(f'\n=== Optimized: axi = {axi_opt:.3f} , alpha = {alpha_opt:.3f} ===')


# ----------------------------------------------------------------------
# 4. 收斂曲線
# ----------------------------------------------------------------------
hist_conc = np.array(hist['conc'])
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
ax[0].plot(hist['loss']); ax[0].set_title('Loss (= -mean log focal concentration)')
ax[0].set_xlabel('iteration'); ax[0].grid(alpha=0.3)
ax[1].plot(hist['axi'], label='axi'); ax[1].plot(hist['alpha'], label='alpha')
ax[1].set_title('Parameters'); ax[1].set_xlabel('iteration'); ax[1].legend(); ax[1].grid(alpha=0.3)
for j, lam in enumerate(wavelengths):
    ax[2].plot(hist_conc[:, j], label=f'{lam*1e6:.1f} µm')
ax[2].set_title('Focal concentration per wavelength'); ax[2].set_xlabel('iteration')
ax[2].legend(); ax[2].grid(alpha=0.3)
plt.tight_layout()
p = os.path.join(OUTDIR, 'axicubic_opt_convergence.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print('saved', p)


# ----------------------------------------------------------------------
# 5. 驗證: 三個波長在 z = 2.4 mm 的聚焦結果
# ----------------------------------------------------------------------
U0_opt = build_field(torch.tensor(axi_opt, device=device),
                     torch.tensor(alpha_opt, device=device)).detach()
x_um = X[Ny // 2, :].detach().cpu().numpy() * 1e6

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
results = {}
for j, lam in enumerate(wavelengths):
    prop = ASMPropagation(propagation_wavelength=lam, propagation_distance=z_focus,
                          n=n_medium, dtype=dtype, device=device)
    prop.set_input_field(U0_opt, pixel_size)
    prop.propagate()
    I = prop.get_intensity.detach().cpu().numpy()
    results[lam] = I

    axm = axes[0, j]
    im = axm.imshow(I, extent=[x_um.min(), x_um.max(), x_um.min(), x_um.max()],
                    cmap='turbo', origin='lower')
    axm.set_title(f'{lam*1e6:.1f} µm  @ z=2.4mm'); axm.set_xlabel('x (µm)'); axm.set_ylabel('y (µm)')
    axm.set_xlim(-150, 150); axm.set_ylim(-150, 150)
    plt.colorbar(im, ax=axm, fraction=0.046)

    line = I[Ny // 2, :]
    line_n = line / line.max()
    axl = axes[1, j]
    axl.plot(x_um, line_n)
    half = line_n >= 0.5
    if half.any():
        xs = x_um[half]
        fwhm = xs.max() - xs.min()
        axl.axhline(0.5, color='r', ls='--', lw=1)
        axl.set_title(f'{lam*1e6:.1f} µm  FWHM ≈ {fwhm:.1f} µm')
    else:
        axl.set_title(f'{lam*1e6:.1f} µm  cross-section')
    axl.set_xlim(-150, 150); axl.set_xlabel('x (µm)'); axl.set_ylabel('norm. I'); axl.grid(alpha=0.3)
plt.tight_layout()
p = os.path.join(OUTDIR, 'axicubic_opt_focus_xy.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print('saved', p)


# ----------------------------------------------------------------------
# 6. x–z 縱向強度: 確認焦點位於 z ≈ 14 mm
# ----------------------------------------------------------------------
# --- 掃描波長設定 (可自訂數量): linspace(min_lam, max_lam, N) ---
min_lam = 8e-6        # 掃描最短波長
max_lam = 12e-6       # 掃描最長波長
N_lam   = 3           # 掃描波長數量 (可改成任意正整數)
scan_wavelengths = np.linspace(min_lam, max_lam, N_lam)

z_range = np.linspace(10e-3, 20.0e-3, 120)

# 依波長數量自動排版子圖 (每列最多 ncols 個)
ncols = min(N_lam, 4)
nrows = int(np.ceil(N_lam / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
axes_flat = axes.ravel()

for j, lam in enumerate(scan_wavelengths):
    prop = ASMPropagation(propagation_wavelength=float(lam), propagation_distance=z_focus,
                          n=n_medium, dtype=dtype, device=device)
    prop.set_input_field(U0_opt, pixel_size)
    prop.propagate_xz(z_range=z_range)
    Ixz = (prop.get_output_UZ.abs() ** 2).detach().cpu().numpy().T   # [Nx, Nz]
    axm = axes_flat[j]
    im = axm.imshow(Ixz, extent=[z_range.min()*1e3, z_range.max()*1e3, x_um.min(), x_um.max()],
                    cmap='turbo', origin='lower', aspect='auto')
    axm.axvline(z_focus*1e3, color='w', ls='--', lw=1)
    axm.set_ylim(-200, 200)
    axm.set_title(f'{lam*1e6:.2f} µm  (x–z)'); axm.set_xlabel('z (mm)'); axm.set_ylabel('x (µm)')
    plt.colorbar(im, ax=axm, fraction=0.046)

# 隱藏多餘的空白子圖
for k in range(N_lam, len(axes_flat)):
    axes_flat[k].axis('off')

plt.tight_layout()
p = os.path.join(OUTDIR, 'axicubic_opt_focus_xz.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'saved {p}  (swept {N_lam} wavelengths: '
      f'{min_lam*1e6:.2f}–{max_lam*1e6:.2f} µm)')


# ----------------------------------------------------------------------
# 7. 儲存優化結果
# ----------------------------------------------------------------------
summary = {
    'axi': axi_opt,
    'alpha': alpha_opt,
    'wavelengths_um': [l * 1e6 for l in wavelengths],
    'z_focus_mm': z_focus * 1e3,
    'lens_D_mm': lens_D * 1e3,
    'pixel_size_um': pixel_size * 1e6,
    'final_concentration': hist['conc'][-1],
    'final_loss': hist['loss'][-1],
}
with open(os.path.join(OUTDIR, 'axicubic_opt_result.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
np.savez(os.path.join(OUTDIR, 'axicubic_opt_history.npz'),
         loss=np.array(hist['loss']), axi=np.array(hist['axi']),
         alpha=np.array(hist['alpha']), conc=np.array(hist['conc']))
print(json.dumps(summary, indent=2, ensure_ascii=False))
