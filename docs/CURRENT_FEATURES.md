# 乾淨版本功能與整理範圍

此文件描述 1.0.7 整理版本。原始專案、私人光源與模擬結果不隨發布收錄。

## 功能盤點

小型 meta-atom 資料庫集中在 data/metaatoms/；範例、notebook、
UI 預設路徑與基本測試均同步更新，不再放在專案根目錄。

- `core/`：不可變 Grid2D / Field2D，標準 `[component,y,x]`，SI 公尺，Ex/Ey。
- `propagation/`：ASM、BLAS、Fresnel、Rayleigh–Sommerfeld，取樣檢查、padding、精度與消逝波政策；保留舊介面。
- `elements.py`、`metaatom.py`：理想相位、光源、Zemax POP 載入、資料庫相位/振幅查表。
- `io/`：NPZ、MAT、ZBF、格式映射與輸入驗證。
- `system/`：光源、薄元件、介面與多層光學系統。
- `project/`：專案儲存、版本與資產管理。
- `gui/`、`cli.py`：桌面工作台、檢視與轉檔指令。
- `tests/`：核心、傳播、I/O、系統、GUI、專案與 CLI 測試。

## 新增 ASM 移動視窗

`propagate_angular_spectrum(field, spec, window_shift_m=(sx,sy))`
使用正號 `exp(i*(kx*sx+ky*sy))`，求原場在移動取樣位置的值。
回傳 Field2D 的 grid center 同步移動，z 為原始 z 加傳播距離。

`scan_moving_window_xz` 使用 `shift_x=offset_x+slope_x*distance`，
回傳複數場與二維全域 x/z 座標。用 pcolormesh 畫全域座標得到
平行四邊形；局部座標顯示矩形。偶數 y 網格不一定含 y=0，
回傳的 y_m 明確標示切片位置。

不支援傾斜觀察面、縮放取樣或移動 BLAS；FFT 位移不會消除週期
副本，必須做 padding/視窗收斂檢查。既有 sampling 報告是中心視窗
估計，不保證移動視窗的頻譜充分取樣。XZ 目前使用完整二維反轉換，
以確保奇偶、偏振與座標一致，尚未採用 ky 加總的效能最佳化。

範例：`python Example/ASM_MovingWindow.py`，不需私人光源。
輸出位於 `output/asm_moving_window/`。大型 Zemax 光源需自行提供，
歷史波長分析範例仍依賴原專案的 source/，不隨乾淨副本發送。

## GitHub 整理方式

本次驗證：386 個測試通過（含 3 個新增移動視窗測試）。使用 torcwa_app
Python 3.12；系統 pytest 暫存目錄受權限限制，改用專案 tmp/ 後通過。
Example/ASM_MovingWindow.py 已執行及目視檢查。
副本打包設定移除未使用的 setuptools_scm，避免從父層倉庫自動收集檔案。

此整理版本作為倉庫根目錄發布，不以 optiprop_clean 巢狀套件提交。
保留 optiprop/、tests/、docs/、Example/、.github/、打包文件與小型資料庫。
排除 output*/、source/、build/、dist*/、快取與生成的 mat/zbf/png。
保留遠端歷史，排除原始大型模擬資料。GUI 移動視窗控制項及舊 ASMPropagation API
尚未接入本次新功能，請使用上述 canonical API。
