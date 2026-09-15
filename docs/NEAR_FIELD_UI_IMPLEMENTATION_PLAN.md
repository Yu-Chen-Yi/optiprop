# OptiProp Near-Field UI 實作規格

狀態：Draft 1  
目標版本：OptiProp 2.0  
GUI：PySide6  
數值後端：PyTorch  
核心單位：SI（m、rad）

目前進度（2026-07-30）：

- Phase 0 測試與 CI 骨架：完成。
- WP-01 Grid2D/Field2D：完成。
- WP-02 Propagator base and sampling report：完成。
- WP-03 ASM/BLAS：完成；包含 coupled Matsushima band limit、CPU/CUDA 與 reference tests。
- WP-04 Fresnel correction：完成；包含 TF/scaled、自然輸出網格與 legacy 非方形修正。
- WP-05 RS-FFT/RS-Direct：完成；包含完整 RS-I 近場項、線性卷積與 chunked direct。
- WP-06 Layer protocol and OpticalSystem：完成；包含 UUID 編輯、逐層結果、驗證與 partial cancellation。
- WP-07 Source/thin-element/interface layers：完成；包含 analytic/imported source factory、exact/paraxial lens、aperture、complex mask 與正常入射介質界面。
- WP-08 Canonical NPZ/MAT：完成；包含 strict canonical schema、inspection、explicit legacy mapping、atomic save 與 NPZ pickle/ZIP 安全邊界。
- 下一工作包：WP-09 ZBF package integration。

## 1. 產品目標

將 OptiProp 從單次傳播函式庫擴充成可重複使用的 near-field 光路設計與分析工具。使用者應能：

1. 建立 analytic incident source，或從 ZBF、MAT、NPZ 匯入複數光場。
2. 直接設定 source 的 amplitude、phase、偏振、波長、位置與角度。
3. 以可拖放排序的 layer 建立任意長度光路。
4. 在 propagation layer 選擇 ASM、band-limited ASM、Fresnel 或 Rayleigh–Sommerfeld。
5. 插入 ideal lens、complex amplitude/phase mask、aperture、介質界面與 meta-atom element。
6. 查看任一 layer 前後的 amplitude、phase、intensity、line profile、XZ map 與功率統計。
7. 保存 project，重開後得到相同的光路、參數與結果定義。
8. 匯出 canonical NPZ、MAT、ZBF 與圖像。
9. 在 CPU/CUDA 上背景運算，支援進度、取消、錯誤定位與取樣警告。

此工具首先是 coherent scalar/vector-component propagation workbench，不是完整 Maxwell solver。Ex/Ey 在 MVP 中獨立傳播；除非使用明確的 interface/Jones layer，系統不自動推導 polarization coupling。

## 2. 名詞與物理語義

### 2.1 ASM、BLAS 與 ASBM

UI 使用下列名稱：

- `ASM`：Angular Spectrum Method。
- `BLAS`：Band-Limited Angular Spectrum Method。
- `Fresnel`：single-FFT Fresnel propagation。
- `RS-FFT`：以 convolution/FFT 實作的 Rayleigh–Sommerfeld propagation。
- `RS-Direct`：小網格、任意 output coordinates 的 reference/advanced mode。

需求中若「ASBM」指 angular-spectrum beam propagation，第一版對應 `ASM`；若是特定論文定義的演算法，必須另外提供公式與 reference，不能默認和 ASM 相同。

### 2.2 均勻介質與界面

- `PropagationLayer`：在固定 refractive index 的均勻介質內傳播距離 `z`。
- `InterfaceLayer`：從 `n1` 跨到 `n2`，可套用 Fresnel/Jones transmission。
- 只改 propagation layer 的 `n` 不代表已模擬反射。
- MVP 可提供 `ignore reflection` 模式，但 UI 必須顯示警告。

### 2.3 Thin element

所有 thin lens、mask、aperture 與 phase plate 使用：

```text
U_out(x, y) = U_in(x, y) * T(x, y)
T(x, y) = A(x, y) * exp(i * phi(x, y))
```

若為 Ex/Ey 場，scalar transmission 預設同時作用於兩個分量；需要 polarization coupling 時使用 2x2 Jones matrix element。

## 3. 非目標

OptiProp 2.0 MVP 不承諾：

- full-wave FDTD/FEM/RCWA。
- 自動處理強散射、多重反射與 standing waves。
- 非均勻折射率體積傳播；這應在後續以 split-step BPM 加入。
- 任意曲面座標系。
- Zemax 所有版本與所有非公開 ZBF variant。
- 超大 3D volume 的即時顯示。

## 4. 分層架構

```text
PySide6 GUI
  ├─ Project/Layer Qt models
  ├─ Property editors
  ├─ Plot views
  └─ Background job controller
          │
Application services
  ├─ ProjectService
  ├─ ImportExportService
  ├─ SimulationService
  └─ ResultCache
          │
Domain/core
  ├─ Field2D / Grid2D
  ├─ OpticalSystem
  ├─ Layer protocol
  ├─ Validation/reporting
  └─ Result/provenance
          │
Numerical implementations
  ├─ ASM / BLAS
  ├─ Fresnel
  ├─ RS-FFT / RS-Direct
  ├─ Sources
  └─ Thin elements/interfaces
          │
PyTorch CPU/CUDA
```

GUI 不得直接包含傳播公式、MAT key 判斷或 ZBF 單位換算。GUI 只操作 domain model 和 application services。

## 5. 建議目錄

```text
optiprop/
  core/
    field.py
    grid.py
    polarization.py
    optical_system.py
    result.py
    validation.py
  layers/
    base.py
    source.py
    propagation.py
    thin_element.py
    interface.py
    observation.py
  propagation/
    base.py
    asm.py
    fresnel.py
    rayleigh_sommerfeld.py
    sampling.py
  io/
    common.py
    zbf.py
    mat.py
    npz.py
    project.py
  services/
    simulation.py
    import_export.py
    result_cache.py
  gui/
    app.py
    main_window.py
    models/
      layer_list_model.py
      project_model.py
    panels/
      source_panel.py
      propagation_panel.py
      element_panel.py
      import_panel.py
    views/
      field_plot.py
      profile_plot.py
      metadata_view.py
    workers/
      simulation_worker.py
  cli.py
tests/
  core/
  propagation/
  io/
  layers/
  gui/
  regression/
```

舊的 public imports 先保留 compatibility wrappers，至少經過一個 minor release 再移除。

## 6. Canonical data model

### 6.1 Grid2D

```python
@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    dx: float
    dy: float
    x_center: float = 0.0
    y_center: float = 0.0
```

規則：

- tensor shape 固定為 `[..., ny, nx]`。
- X 是最後一個 axis，Y 是倒數第二個 axis。
- `dx`、`dy` 使用 meter 且必須大於零。
- coordinate 定義必須對 odd/even size 一致並有測試。
- 不再把 `Nx/Ny` 同時當作 tensor row/column。

### 6.2 Field2D

```python
@dataclass
class Field2D:
    data: torch.Tensor
    grid: Grid2D
    wavelength_m: float
    medium_index: complex = 1.0
    components: tuple[str, ...] = ("scalar",)
    z_m: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

canonical shape 固定為：

- scalar：`[1, ny, nx]`
- polarized：`[2, ny, nx]`，component 固定 `("Ex", "Ey")`
- batch/multi-wavelength 不在 data axis 中暗示；以明確的 batch request/result 表示。

驗證：

- complex64 或 complex128。
- shape 與 grid 完全一致。
- wavelength > 0。
- medium index 的 real part > 0。
- 不允許 silent transpose、silent unit guessing 或 silent resampling。

座標採：

```text
x = (arange(nx) - (nx - 1) / 2) * dx + x_center
y = (arange(ny) - (ny - 1) / 2) * dy + y_center
```

ZBF 若採 `(index - N/2) * d`，匯入後以 `x_center=-dx/2`、`y_center=-dy/2` 保留原樣本座標，不可靜默 recenter。

相位 convention 固定為 `exp(-iωt)`，正 z outgoing wave 為 `exp(+ikz)`，並寫入 project schema。

衍生量：

- `amplitude`
- `phase(wrapped=True)`
- `intensity(component=None)`
- `power()`，需明確採用離散積分與 normalization convention。
- `to(device, dtype)`
- `clone_with(...)`

### 6.3 FieldResult

每一層輸出需包含：

```python
@dataclass
class LayerResult:
    layer_id: UUID
    input_field: Field2D | FieldSummary
    output_field: Field2D
    metrics: dict[str, float]
    warnings: list[ValidationMessage]
    elapsed_s: float
    cache_key: str
```

記憶體不足時，非選取層可只保留 `FieldSummary` 或磁碟 cache。

## 7. Layer model

### 7.1 共通 protocol

```python
class OpticalLayer(Protocol):
    id: UUID
    name: str
    enabled: bool

    def validate(self, field: Field2D | None) -> ValidationReport: ...
    def apply(self, field: Field2D, context: RunContext) -> Field2D: ...
    def to_config(self) -> dict: ...
```

所有 layer 都必須：

- 有穩定 UUID。
- 可 JSON serialization。
- validation 不修改 field。
- 執行不修改 input tensor。
- 參數變更時能產生 deterministic cache key。

### 7.2 IncidentSource

`IncidentSource` 是獨立、不可變的 source factory，由 Source panel 設定並
以 `create()` 產生送入 `OpticalSystem` 的初始 `Field2D`。它不是
field-in/field-out `OpticalLayer`，因此不能插在光路中途覆寫上游結果；
後續 MAT/NPZ/ZBF importer 也使用相同的 source 邊界。

種類：

- Plane wave
- Tilted plane wave
- Gaussian beam
- Elliptical Gaussian beam
- Imported complex field
- Imported amplitude + phase

共通參數：

- wavelength
- polarization/components
- amplitude
- phase offset
- source center
- aperture
- grid

imported source 另外保存 import mapping 和 provenance。

### 7.3 PropagationLayer

參數：

- method
- distance
- refractive index
- padding mode/size/factor
- crop policy
- evanescent policy
- band-limit policy
- output sampling policy
- precision override（optional）
- keep intermediate result

### 7.4 IdealLensLayer

參數：

- focal length
- design wavelength
- clear aperture
- center
- transmission amplitude
- phase offset
- phase sign convention
- medium index

phase sign 必須以測試確認正焦距在正 z 方向聚焦。

### 7.5 ComplexMaskLayer

輸入可為：

- complex field
- amplitude + phase
- amplitude only
- phase only

如果 mask grid 與 field 不同，預設拒絕執行；使用者必須明確選擇：

- resample complex real/imag
- resample amplitude + unwrapped phase
- crop/pad

每種策略都需顯示預覽。

### 7.6 ApertureLayer

- circle
- rectangle
- ellipse
- imported binary/continuous mask
- center、rotation、edge convention

### 7.7 InterfaceLayer

MVP：

- `n1`、`n2`
- normal incidence scalar transmission
- ignore reflection switch

進階：

- incident direction
- s/p Fresnel coefficients
- Jones matrix
- complex refractive index

### 7.8 MetaAtomLayer

包裝現有 MetaAtomLibrary/MetaAtomElement：

- library path 或 embedded project reference
- wavelength selection/interpolation policy
- alpha
- global phase offset optimization
- aperture
- parameter map export

### 7.9 ObservationLayer

不改變 field，只控制：

- checkpoint
- metrics
- export
- XZ scan request

## 8. Propagation engine

### 8.1 共通介面

```python
class Propagator(ABC):
    def validate(
        self, field: Field2D, distance_m: float, options: PropagationOptions
    ) -> ValidationReport: ...

    def propagate(
        self,
        field: Field2D,
        distance_m: float,
        options: PropagationOptions,
        cancel_token: CancelToken | None = None,
    ) -> Field2D: ...
```

### 8.2 ASM

必須明確處理：

- frequency grid 使用各自 `dx/dy`。
- propagating 與 evanescent components。
- positive/negative z。
- padding/crop。
- complex square root branch。
- FFT normalization。
- 頻率軸必須對應 tensor 最後兩維 `[y, x]`，不能把 row index 命名成 X。

evanescent policy：

- `discard`：超出 propagating circle 設為零。
- `decay`：保留正 z 的衰減。
- `keep`：advanced，需警告 backward propagation 可能爆增。

`kz=sqrt(k²-kx²-ky²)` 必須選擇 `Re(kz)>=0`、`Im(kz)>=0` 的 branch。預設先限制為均勻、實數 refractive index；複數 n 要有獨立 absorption 測試後才開放。

### 8.3 BLAS

Band limit 應由 sampling、distance、wavelength 與 padded extent 推導，不以固定 magic number 實作。Validation report 應呈現：

- 是否有 wrap-around 風險。
- 建議 padding。
- 被 band-limit 移除的 spectrum 比例。

### 8.4 Fresnel

分成兩種 numerical path：

- `fresnel_tf`：same-grid transfer function。
- `fresnel_scaled`：single-FFT scaled Fresnel，output grid 由公式決定。

需要修正與測試：

- output sampling：`dx_out = lambda_medium * abs(z) / (nx_fft * dx_in)`。
- X/Y 分別計算。
- z=0 明確 bypass 或拒絕。
- output coordinate ordering。
- padding 對 output sampling 的影響。
- global phase convention。
- `ifftshift → fft2 → fftshift` 與 output coordinate 必須成對一致。

UI 必須顯示 Fresnel output grid 可能和 input grid 不同。

### 8.5 RS-FFT

作為一般 UI 的 RS 選項：

- exact RS-I convolution kernel；在 `exp(-iωt)` convention 下包含近場項：

```text
h = z / (i λ r²) * exp(i k r) * (1 - 1 / (i k r))
```

- linear convolution padding，避免 circular convolution。
- crop 回 requested output。
- sampling validity report。
- CPU/GPU compatible。
- MVP 限制 output `dx/dy` 與 input 相同且座標 aligned；任意 spacing 交給 RS-Direct。

### 8.6 RS-Direct

只用於：

- 小 grid reference。
- 任意 output positions。
- regression comparison。

要求：

- chunking，避免建立完整 `[Nout, Nin]` tensor。
- 顯示 operation/memory estimate。
- 超過安全 threshold 時要求使用者確認或拒絕。
- 每個 chunk 檢查 cancel token。

### 8.7 多分量與多波長

- Scalar、Ex、Ey 共用相同 spatial propagator，可把 component 視為 leading batch axis。
- Multi-wavelength run 使用 `SimulationRequest.wavelengths` 逐波長或批次執行。
- 不允許拿單一 wavelength 的 imported phase 默認套用其他 wavelength；必須選擇 reuse、scale optical path 或指定 wavelength-dependent data。

### 8.8 可微分性

- GUI 執行預設使用 `torch.inference_mode()`。
- library API 必須保留 differentiable path，避免破壞既有 inverse-design。
- 當 input/spec `requires_grad` 時不得使用 detached transfer-function cache。
- 加入 autograd smoke test 與小型 finite-difference comparison。

## 9. Sampling validation

每次 run 前先產生 ValidationReport：

- grid Ny/Nx、dx/dy、physical extent。
- wavelength/medium wavelength。
- propagation distance。
- estimated Fresnel number。
- angular-spectrum propagating bandwidth。
- aliasing/wrap-around risk。
- recommended padding。
- estimated GPU/CPU memory。
- RS-Direct estimated operations。
- mask/source extent mismatch。

訊息 severity：

- info
- warning：可執行但可能不準確。
- error：禁止執行。

Validation 必須能定位到 layer UUID，GUI 點擊訊息後可選到該 layer。

## 10. I/O 規格

### 10.0 現有資料的 migration 風險

- 現有 MAT/NPZ 沒有一致的 key、單位、axis 或 plane metadata。
- `Laser2Metalens_Collimator_MetaAtom.py` 曾把 `lambda` 寫成 `DESIGN_LAMBDA * 1e3`，但其他欄位仍用 meter；`IdealPhase` 又直接保存 meter，兩者相差 1000 倍。
- wavelength-analysis MAT 只保存 EX/EY/dx/dy/Nx/Ny，缺少 wavelength、medium index、z、orientation 與 normalization。
- 這些 legacy 檔案只能由 importer 提出 mapping 建議；單位不明時必須標為 ambiguous，不能用數值大小靜默猜測。

### 10.0.1 共通 API

```python
inspect_field(path) -> DatasetInspection
load_field(path, mapping=None, *, strict=True) -> Field2D
save_field(field, path, *, format=None, options=None) -> ExportReport
```

`DatasetInspection` 至少包含 key inventory、shape、dtype、complex flag、min/max、metadata candidates、mapping confidence 與 warnings。匯入器先 inspect，再由 explicit `ImportMapping` load。

### 10.1 Canonical NPZ

新輸出固定 keys：

```text
schema_name       = "optiprop.field2d"
schema_version    = 1
field             = complex [C, ny, nx]，C=1 或 2
components        = ["scalar"] or ["Ex", "Ey"]
axis_order         = "C,Y,X"
coordinate_convention = "sample_center_increasing_xy"
field_representation = "relative_electric_field_complex_amplitude"
time_convention    = "exp(-i*omega*t)"
dx_m
dy_m
wavelength_m
medium_index_real
medium_index_imag
x_center_m
y_center_m
z_m
metadata_json
```

不得使用 pickle 保存必要資料。

NPZ importer 固定使用 `allow_pickle=False`；現有 meta-atom database 的 pickle-based 格式不得擴散成通用 field 格式。

### 10.2 MAT

canonical keys 與 NPZ 相同。Importer 同時支援 legacy aliases：

- `EX`、`EY`
- `Ex`、`Ey`
- `U`、`field`
- `U_after_Ex`、`U_after_Ey`
- `dx`、`dy`、`pixel_size`
- `Nx/nx`、`Ny/ny`
- `lambda/wavelength/design_lambda`

自動偵測只能提出 mapping 建議，若 wavelength/unit 缺失仍需使用者確認。

- MAT v4/v5 至 v7.2 使用 SciPy。
- MAT v7.3/HDF5 需要 MATLAB golden fixture 驗證 complex layout 與 axis 後再啟用；WP-08 目前回傳明確 unsupported-format error，不做 blanket transpose。
- MATLAB `(1,1)` scalar metadata 要正規化為 Python scalar。

### 10.3 ZBF

將根目錄 `zbf.py` 整合為 `optiprop.io.zbf`：

- 保留 read/write round-trip。
- 單位在 importer 邊界轉成 SI。
- ZBF array orientation 轉成 canonical `[ny, nx]`。
- ispol=0 對應 scalar/Ex；ispol=1 對應 Ex/Ey。
- 保留 pilot rays 與原始 header 至 metadata。
- export power-of-two padding 改為 option，不散落在 example script。
- strict reader 在配置 array 前先驗證 version、nx/ny、ispol、unit、finite spacing/wavelength/index 與 expected byte size。
- strict 模式拒絕截斷、未知 version 與 trailing bytes；lenient 模式只能轉成 structured warning。
- writer 採同目錄 temporary file、flush/fsync、`os.replace` 的 atomic write。
- `ispol=0` 卻提供 Ey、或 `ispol=1` 缺 Ey，必須報錯。
- 除自寫自讀外，需加入真正 OpticStudio v0/v1 golden fixtures。

### 10.4 Import mapping dialog

顯示：

- 所有 array key、shape、dtype、min/max。
- complex field 或 amplitude/phase mapping。
- amplitude/intensity 選擇。
- rad/degree。
- Ex/Ey mapping。
- length/wavelength unit。
- transpose、flip X/Y。
- crop/pad/resample。
- 匯入前 amplitude/phase preview。

使用者選擇需保存於 project，重開時可重現。

WP-08 的 explicit mapping 支援 1/2 component complex arrays，以及
amplitude/intensity + phase（rad/deg）、YX/XY、flip X/Y。Real/imag 分離
arrays、任意 3-D component axis 與 implicit singleton squeeze 尚未支援；
遇到這些資料必須明確拒絕，不能自行猜測。

## 11. Project schema

專案副檔名使用 `.opproj`，內容為 JSON Schema Draft 2020-12。第一層至少包含：

```json
{
  "schema_name": "optiprop.project",
  "schema_version": 1,
  "application_version": "2.0.0",
  "project_id": "stable-uuid",
  "name": "Example multilayer system",
  "default_compute": {
    "device": "auto",
    "precision": "float32"
  },
  "sources": [],
  "layers": [],
  "assets": {},
  "runs": [],
  "ui_state": {}
}
```

要求：

- schema version migration。
- 相對檔案路徑優先。
- asset 使用 stable ID、相對路徑與 SHA-256；外部絕對路徑標示 non-portable。
- missing file relink。
- 不把大型 tensor 直接塞進 JSON。
- 可選擇 copy imported assets into project directory。
- unknown future fields 應保留或提出明確錯誤。
- project save 採 atomic write。
- optional portable bundle 使用獨立副檔名，解包時防止 path traversal。
- intermediate result 不寫入 project JSON，只寫 cache manifest。
- local UI preferences 與可攜 project 分離。

## 12. Simulation execution

### 12.1 執行模式

- Validate only
- Run all
- Run to selected layer
- Run selected layer onward
- Recompute invalidated layers
- Batch wavelengths
- XZ scan

### 12.2 Cache invalidation

cache key 包含：

- input field hash/identity。
- layer type/version。
- normalized parameters。
- backend/precision。
- relevant external file checksum。

修改第 N 層只 invalidate N 及之後的結果。

### 12.3 Cancel

取消是 cooperative：

- 每層之間檢查。
- multi-z/multi-wavelength 每次 iteration 檢查。
- RS-Direct 每個 chunk 檢查。
- 單次 FFT 無法中途中止，GUI 顯示「正在完成目前 kernel」。

## 13. PySide6 UI

### 13.1 主視窗

```text
┌ Toolbar: New Open Save | Validate Run Stop | Device Precision ┐
├ Source/Library ┬ Optical System Layers ┬ Properties/Validation ┤
│ source presets │ ordered drag/drop list│ selected layer editor │
│ element presets│ status/cache/time     │ warnings and estimates│
├────────────────┴────────────────────────┴───────────────────────┤
│ Results tabs: Field | Profiles | XZ | Metrics | Metadata | Log │
└ Status bar: progress, current layer, memory/device              ┘
```

### 13.2 Layer list

使用 `QAbstractListModel` 或 `QAbstractTableModel`，不可只靠 widget item 保存 domain state。

操作：

- add
- delete
- duplicate
- enable/disable
- drag/drop reorder
- rename
- multi-select
- run to here
- export this plane

所有 add/delete/duplicate/move/enable/parameter edit 都走 `QUndoStack` command；選取狀態以 layer UUID 維持，不能依賴會因 reorder 改變的 row number。

每一列顯示：

- icon/type。
- name。
- enabled。
- validation severity。
- cache state。
- elapsed time。

### 13.3 Property editor

使用 editor registry 與 `QStackedWidget`，依 layer type 切換 panel，避免在 MainWindow 堆疊大量 type 判斷。數值欄使用：

- `QDoubleSpinBox`/scientific notation widget。
- 明確 unit selector。
- validation state。
- tooltip 顯示物理意義與限制。

編輯先進入 project model；不應每打一個字就啟動完整 simulation。採 debounce 或 Apply/auto-run option。

### 13.4 Result view

選取 layer 與 plane（before/after），可看：

- amplitude。
- wrapped/unwrapped phase。
- intensity。
- real/imag。
- Ex/Ey/total。
- X/Y line cuts。
- XZ intensity。
- metrics。
- metadata/provenance。

色階支援：

- auto/fixed/shared。
- linear/log intensity。
- phase 固定 `[-pi, pi]`。
- phase 可依 amplitude threshold 隱藏低訊號區域。
- cursor 座標與值。
- zoom/pan/reset。
- worker signal 只傳 `ResultHandle`；大型 CPU/GPU tensor 由 `ResultStore` 管理。
- UI 可先使用 decimated CPU preview，export 時才取 full-resolution result。

### 13.5 Worker

建議：

- GUI controller 建立 immutable SimulationRequest。
- `QObject` worker 移至 `QThread`，或使用 `QThreadPool/QRunnable`。
- signals：started、progress、layer_finished、warning、failed、cancelled、finished。
- worker 不直接操作 QWidget。
- exception 需保留 traceback 到 log，對使用者顯示可理解摘要。
- 禁止使用 `QThread.terminate()`；取消只透過 thread-safe cancellation token。
- 關閉正在運算的 project 時，先 cooperative cancel，再處理退出。
- project 修改、執行與錯誤狀態使用明確 state machine：idle、preflight、running、cancelling、failed、complete。

## 14. Packaging

建議：

- Python >= 3.10。
- `PySide6` 和 GUI plotting dependency 放在 `[project.optional-dependencies].gui`。
- setuptools 改用 package discovery，確保 `optiprop.core/io/gui/...` 子套件會被安裝。
- `pyproject.toml` 成為唯一 package metadata 來源；version 只由一個 dynamic source 取得。
- `setup.py` 移除或只保留極薄 compatibility shim。
- console scripts：

```toml
[project.scripts]
optiprop = "optiprop.cli:main"
optiprop-gui = "optiprop.gui.app:main"
optiprop-inspect = "optiprop.io.cli:inspect_main"
optiprop-convert = "optiprop.io.cli:convert_main"
```

- README 提供 core-only 與 GUI 安裝。
- Windows 可另外建立 launcher，但不能把 `.bat` 當唯一入口。
- CUDA PyTorch 安裝依官方 wheel 指令說明，不使用無效的 `torch[cuda]` extra。
- 增加 Windows、Linux 的 core CI；GUI 至少做 offscreen smoke test。
- wheel/sdist 驗證必須確認子套件、schema、icon 與 example project 被正確包含，且不含 output、個人設定或大型研究產物。

## 15. 測試策略

### 15.1 Core

- Grid odd/even coordinate convention。
- rectangular grid。
- Field shape/component validation。
- device/dtype conversion。
- power calculation。

### 15.2 Propagation

- plane wave phase advance。
- Gaussian beam analytic waist/curvature。
- 非方形 `127x192`、`dx != dy` 與 odd/even grid。
- ASM forward/backward round-trip（在 band-limited field 上）。
- ASM/BLAS agreement in safe sampling region。
- Fresnel output sampling。
- Fresnel/ASM paraxial agreement。
- RS-FFT/RS-Direct small-grid agreement。
- exact RS-I 近場項 regression。
- rectangular dx/dy。
- zero/negative distance。
- evanescent policies。
- power conservation tolerance。

### 15.3 Elements/layers

- lens focus sign。
- amplitude/phase mask multiplication。
- aperture geometry。
- disabled layer identity。
- Ex/Ey independent application。
- Interface coefficients。
- multilayer result 等於手動逐層組合。

### 15.4 I/O

- canonical NPZ round-trip。
- MAT round-trip。
- legacy MAT aliases。
- ZBF version/isPol/unit/orientation round-trip。
- missing metadata。
- invalid key/shape。
- transpose/flip mapping。
- mixed-unit/ambiguous legacy MAT 必須拒絕自動匯入。
- NPZ object/pickle 必須拒絕。
- project JSON round-trip/migration/missing files。
- asset checksum mismatch、duplicate UUID、atomic save。

### 15.5 GUI

- layer add/delete/duplicate/reorder。
- property edit 更新 model。
- validation message 選取正確 layer。
- worker finished/failed/cancelled。
- project open/save。
- undo/redo 與 layer UUID/selection 在 reorder 後保持。
- headless smoke test。
- 大型匯入背景執行與取消。

### 15.6 Regression

選擇小型固定案例保存 reference metrics，不保存過大的完整場：

- Air propagation。
- Glue → lens → air。
- Imported Ex/Ey → mask → propagation。
- Multi-wavelength run。

CPU complex128 reference 產生 baseline；CPU/GPU 使用明確的 `rtol/atol`。

## 16. 實作分期

### Phase 0：基準與保護

- 建立 feature branch。
- 記錄 dirty worktree，不刪除既有 output。
- 不提交個人 `.wavelength_analysis_ui.json` 內的絕對路徑。
- 把目前可執行案例縮小成 regression fixture。
- 建立真正的 `tests/` 與 CI skeleton。
- 先加入會暴露目前 X/Y、Fresnel sampling、RS kernel 問題的 failing reference tests。

完成條件：舊 API smoke test 可跑，既有檔案無遺失。

### Phase 1：Field/Grid 與 propagation 修正

- Grid2D/Field2D。
- common Propagator API。
- ASM、BLAS、Fresnel。
- sampling validation。
- compatibility wrappers。

完成條件：analytic/reference tests 通過，rectangular grid 可用。

### Phase 2：RS 與 multilayer engine

- RS-FFT。
- chunked RS-Direct。
- layer model。
- OpticalSystem execution/cache invalidation。
- scalar/Ex/Ey。

完成條件：任意數量 layer 可由 headless Python API 執行並取得每層結果。

### Phase 3：I/O 與 project

- ZBF 整合。
- canonical MAT/NPZ。
- legacy mapping。
- versioned project JSON。

完成條件：三種 field 格式 round-trip，project 重開結果定義一致。

### Phase 4：PySide6 MVP

- Main window。
- Layer model/editor。
- Source/import panels。
- background worker。
- amplitude/phase/intensity/metadata。
- open/save/run/stop。

完成條件：不用寫 Python 即可完成 source → propagation → lens → propagation 並查看任一 plane。

### Phase 5：分析與品質

- profiles、XZ、metrics。
- batch wavelengths。
- cache UI。
- warnings/memory estimate。
- docs、example projects、packaging。

完成條件：通過完整 acceptance test，能由 `optiprop-gui` 啟動。

### Phase 6：進階功能

- Interface s/p/Jones。
- dispersion table。
- complex refractive index。
- split-step BPM。
- optimization/inverse design integration。

## 17. 開發工作包

每個工作包應形成獨立、可測試的變更：

1. `WP-01 Grid2D/Field2D`
2. `WP-02 Propagator base and sampling report`
3. `WP-03 ASM/BLAS`
4. `WP-04 Fresnel correction`
5. `WP-05 RS-FFT/RS-Direct`
6. `WP-06 Layer protocol and OpticalSystem`
7. `WP-07 Source/thin-element/interface layers`
8. `WP-08 Canonical NPZ/MAT`
9. `WP-09 ZBF package integration`
10. `WP-10 Project schema`
11. `WP-11 PySide project/layer models`
12. `WP-12 Property editors/import mapping`
13. `WP-13 Simulation worker/cache`
14. `WP-14 Result plots/metrics/XZ`
15. `WP-15 Packaging/docs/regression`

依賴順序：

```text
WP-01 → WP-02 → WP-03/WP-04/WP-05
                     ↓
                  WP-06 → WP-07
WP-01 → WP-08/WP-09 → WP-10
                  WP-06/WP-10 → WP-11/WP-13
                                   ↓
                                WP-12/WP-14
                                   ↓
                                  WP-15
```

不得在 WP-01/02 的 Field/Grid 與 sampling contract 尚未通過 reference tests 前開始把舊 propagator 接進 GUI。

每個 WP 必須包含：

- public behavior。
- unit tests。
- migration/compatibility note。
- example 或 fixture。
- completion checklist。

## 18. MVP 驗收情境

### Scenario A：Analytic source

1. 建立 Gaussian source。
2. 設定 amplitude、phase、waist、wavelength。
3. 新增 20 µm、n=1.45 ASM propagation。
4. 新增 amplitude=0.9 的 ideal lens。
5. 新增 500 µm air BLAS propagation。
6. 執行並查看所有 layer 前後 amplitude/phase/intensity。
7. 保存 project，重開後參數一致。

### Scenario B：Imported polarized field

1. 匯入 ZBF。
2. 正確顯示 Ex/Ey、dx/dy、wavelength、index。
3. 通過 imported complex mask。
4. 使用 Fresnel 或 ASM 傳播。
5. 分別查看 Ex/Ey/total intensity。
6. 匯出 MAT、NPZ、ZBF 並 round-trip。

### Scenario C：Legacy MAT/NPZ

1. 載入目前 Example 產生的 MAT/NPZ。
2. Import dialog 建議正確 key mapping。
3. 使用者確認單位/orientation。
4. 預覽與匯入後 field 一致。

### Scenario D：錯誤與取消

1. 建立明顯 aliasing 的設定。
2. Run 前顯示有 layer 定位的警告。
3. 啟動 multi-z 或 RS-Direct。
4. Stop 後 UI 不凍結，保留先前有效結果。

## 19. Definition of Done

只有同時滿足以下條件才算完成：

- GUI 可以建立並執行任意數量的 layer。
- ZBF/MAT/NPZ 可匯入、預覽、映射與匯出。
- source amplitude/phase 可由 analytic 參數或 map 定義。
- ASM/BLAS/Fresnel/RS-FFT 有 reference tests。
- scalar 與 Ex/Ey 可完整通過 pipeline。
- 每層前後結果可查看。
- validation 能阻止無效設定並警告不可靠取樣。
- 運算在背景執行且可以取消。
- project 可保存、重開與 relink external files。
- 所有核心與 GUI smoke tests 通過。
- 舊 public API 有 compatibility path。
- README、example project、安裝與啟動方式完整。

## 20. 實作時的保護規則

- 不刪除 repository 內既有 output、研究資料或未提交檔案。
- 不在 Example script 中繼續複製新的 I/O helper；功能應進入 package。
- 不用 interpolation 掩蓋 grid mismatch，所有 resampling 都需明確選擇。
- 不把 Matplotlib `show()` 放進 numerical core。
- 不讓 GUI widget 成為 project 的唯一資料來源。
- 不以 class 名稱宣稱演算法正確；每個方法都需要 reference test。
- 不在沒有單位 metadata 時靜默猜測。
- 每次改動只 invalidate 受影響 layer 與 downstream cache。
