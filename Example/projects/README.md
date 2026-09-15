# 可載入的專案 JSON

這些範例使用現有 canonical `optiprop-project` schema v1，並非準直專用格式。
所有長度以 m 儲存，UI 可以用 µm/nm 顯示。範例不含大型光場檔。

- `free_propagation.json`：解析 Gaussian Ex/Ey，在空氣中依序傳遞 100 µm、200 µm，沒有透鏡。
- `metalens_focus_ideal.json`：1310 nm 平面波 → Ø60 µm、f=100 µm 的理想相位
  Metalens → ASM 在空氣中傳遞 100 µm 至設計焦平面。512×512、dx=dy=0.25 µm。
  Ex/Ey 共用 scalar phase，入射振幅分別為 1/0.5；不使用 meta-atom 資料庫。
- `metalens_focus_binary2.json`：相同光源、孔徑及 100 µm 傳遞，換成 Binary2 的
  r²、r⁴、r⁶、r⁸ 四項相位；係數來自上例等光程相位的 Taylor 近似，不是精確轉換。
- `laser_collimator_demo.json`：解析 Gaussian → 65 µm 膠水 → 忽略反射的介面 →
  空氣中參考焦長 26.6 µm 的理想相位 → 0.2 µm 空氣傳遞。

準直示範只示範可編輯的光路，**不是 P3_10um 實際量測光源，也不是已驗證的準直設計**。
私人 Ex/Ey 需經正式匯入、建立真實資產引用後另存專案，不能由範例推斷模擬成效。
它不載入 meta-atom 資料庫，也不聲稱可重現既有 Example 的最佳化結果。

目前可透過既有 Python project API 載入與計算；舊 Qt 表單的格式相容性尚未接通，
新通用編輯器仍是操作草稿，不能當成完整 GUI 已完成。

在 repository root 執行：

```python
import torch
from optiprop.project import load_project, project_to_domain

project = load_project("Example/projects/free_propagation.json")
source, system = project_to_domain(project)
# Explicit precision/device: project_to_domain reconstructs objects but does not
# execute compute_settings by itself.
input_field = source.create(dtype=torch.complex128, device="cpu")
result = system.execute(input_field)
output_field = result.output_field
```

內建範例與使用者專案應走相同 loader。監測器/最佳化任務等新增功能需正式
schema 與 runner 支援後再序列化，不以 metadata 冒充已可執行功能。

## 實際計算 Metalens 聚焦

執行 `python Example/Metalens_Focus_Project.py`，會載入 `metalens_focus_ideal.json`，
計算設計焦平面的 Ex/Ey、0–150 µm 的 XZ 掃描，並輸出至
`output/metalens_focus/<timestamp>/`，不覆蓋之前結果：

- `focus_intensity.png`：Ex/Ey 焦平面強度（共用色階）。
- `focal_amplitude_phase.png`：光源通過透鏡後、設計焦平面的 Ex/Ey 振幅/相位。
- `xz_intensity.png`：Ex/Ey 並排 XZ intensity（共用對數色階）。
- `metrics.json`、`focus_scan.npz`：焦點掃描、FWHM、實際切片座標與 padding 檢查。

圖中設計焦長與掃描所得強度最大位置分開報告；有限孔徑的軸向最大值不必恰好等於 f。
掃描是此 Python 範例的後處理，不是 JSON 中未實作的監測器功能。
加上 `--phase-model binary2` 即載入 Binary2 聚焦範例，輸出相同圖與指標。
此診斷腳本只接受一片理想/Binary2 透鏡接一段傳遞；通用光路仍由 project API 執行。

## 理想 / Binary2 選擇

新版操作草稿在同一個「透鏡相位類型」下切換，JSON 分別使用
`IdealLensLayer` / `Binary2LensLayer`。Ex/Ey 共用 scalar transmission。
共同孔徑及元件 ID 不變；切換不會改下游傳遞距離。未啟用模型的設定存入
`ui_state.lens_model_drafts`（Python 不執行 UI 草稿），切回時可還原。

Binary2 沿用舊 `Binary2Phase` 定義：

`phi = phase_offset_rad + C1*r_mm**2 + C2*r_mm**4 + ...`

中心/孔徑仍用 m；半徑計算轉為 mm，Ci 單位 **rad/mm^(2i)**。
不是兩階量化相位、waves 或歸一化半徑多項式。係數指定固定弧度相位，
**改波長不會自動重算係數**；多波長的真實柱子響應仍需資料庫。
手動初次切換預設全零（平坦相位），不把理想焦長偷偷轉成近似係數。
聚焦範例的非零係數是明確的四項 Taylor 範本，可直接修改。
