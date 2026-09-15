"""Desktop UI for Laser2Metalens_Collimator_WavelengthAnalysis.py."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "Metalens 波長分析與 ZBF 匯出工具"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ANALYSIS_SCRIPT = HERE / "Laser2Metalens_Collimator_WavelengthAnalysis.py"
SETTINGS_FILE = HERE / ".wavelength_analysis_ui.json"

COLORS = {
    "navy": "#10233F",
    "blue": "#176BCE",
    "blue_hover": "#0F5CB7",
    "cyan": "#DDF4FF",
    "bg": "#F3F6FA",
    "card": "#FFFFFF",
    "text": "#172033",
    "muted": "#68758A",
    "line": "#DCE3EC",
    "green": "#17875B",
    "red": "#C13B42",
}


def find_analysis_python():
    """Prefer a local Python environment that visibly contains required packages."""
    candidates = [Path(sys.executable)]
    env_roots = []
    if os.environ.get("CONDA_ENVS_PATH"):
        env_roots.extend(Path(item) for item in os.environ["CONDA_ENVS_PATH"].split(os.pathsep))
    env_roots.extend((
        Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".conda" / "envs",
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "miniconda3" / "envs",
    ))
    for root in env_roots:
        if root.is_dir():
            candidates.extend(path / "python.exe" for path in sorted(root.iterdir()))
    seen = set()
    for executable in candidates:
        key = str(executable).lower()
        if key in seen or not executable.is_file():
            continue
        seen.add(key)
        package_dir = executable.parent / "Lib" / "site-packages"
        if all(
            (package_dir / name).exists()
            for name in ("torch", "numpy", "scipy", "matplotlib", "skimage")
        ):
            return str(executable)
    return sys.executable


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Page.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.window, width=event.width),
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if self.winfo_containing(event.x_root, event.y_root) in self.winfo_children_recursive():
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def winfo_children_recursive(self):
        result = [self]
        pending = list(self.winfo_children())
        while pending:
            widget = pending.pop()
            result.append(widget)
            pending.extend(widget.winfo_children())
        return result


class MetalensAnalysisUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1240x800")
        self.minsize(1040, 680)
        self.configure(bg=COLORS["bg"])
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_image = None
        self.preview_paths: dict[str, Path] = {}

        self.vars = self._make_vars()
        self._configure_styles()
        self._build_ui()
        self._load_settings()
        self.after(100, self._poll_process_output)
        self.after_idle(self.refresh_results)

    def _make_vars(self):
        return {
            "python": tk.StringVar(value=find_analysis_python()),
            "source_dir": tk.StringVar(value=str(REPO_ROOT / "source")),
            "library_dir": tk.StringVar(value=str(REPO_ROOT / "data" / "metaatoms")),
            "output_dir": tk.StringVar(value=str(REPO_ROOT / "output")),
            "design_wavelength": tk.StringVar(value="1310"),
            "wavelengths": tk.StringVar(value="1290, 1310, 1330"),
            "min_transmission": tk.StringVar(value="0.80"),
            "glue_index": tk.StringVar(value="1.5"),
            "glue_distance": tk.StringVar(value="125"),
            "lens_diameter": tk.StringVar(value="90"),
            "after_lens": tk.StringVar(value="0.2"),
            "pixel_size": tk.StringVar(value="325"),
            "field_size": tk.StringVar(value="160"),
            "alpha": tk.StringVar(value="0.3"),
            "device": tk.StringVar(value="auto"),
            "z_max": tk.StringVar(value="1000"),
            "z_points": tk.StringVar(value="11"),
            "xz_points": tk.StringVar(value="201"),
            "pad_points": tk.StringVar(value="985"),
            "export_zbf": tk.BooleanVar(value=True),
            "export_mat": tk.BooleanVar(value=True),
            "export_plots": tk.BooleanVar(value=True),
            "status": tk.StringVar(value="待命"),
            "preview_choice": tk.StringVar(value=""),
        }

    def _configure_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.option_add("*Font", ("Microsoft JhengHei UI", 10))
        style.configure("Page.TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure(
            "Header.TLabel",
            background=COLORS["navy"],
            foreground="white",
            font=("Microsoft JhengHei UI", 19, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background=COLORS["navy"],
            foreground="#C6D6EB",
            font=("Microsoft JhengHei UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=COLORS["card"],
            foreground=COLORS["text"],
            font=("Microsoft JhengHei UI", 12, "bold"),
        )
        style.configure(
            "Field.TLabel",
            background=COLORS["card"],
            foreground=COLORS["muted"],
            font=("Microsoft JhengHei UI", 9),
        )
        style.configure(
            "Hint.TLabel",
            background=COLORS["card"],
            foreground=COLORS["muted"],
            font=("Microsoft JhengHei UI", 8),
        )
        style.configure(
            "Primary.TButton",
            background=COLORS["blue"],
            foreground="white",
            borderwidth=0,
            padding=(18, 10),
            font=("Microsoft JhengHei UI", 11, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLORS["blue_hover"]), ("disabled", "#9DB4CF")],
        )
        style.configure(
            "Secondary.TButton",
            background="#E9EFF6",
            foreground=COLORS["text"],
            borderwidth=0,
            padding=(12, 8),
        )
        style.map("Secondary.TButton", background=[("active", "#DCE6F1")])
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Microsoft JhengHei UI", 10))
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["card"])],
            foreground=[("selected", COLORS["blue"])],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=COLORS["blue"],
            troughcolor="#D8E2EF",
            borderwidth=0,
        )

    def _build_ui(self):
        header = ttk.Frame(self, style="Card.TFrame")
        header.configure(height=92)
        header.pack(fill="x")
        banner = tk.Frame(header, bg=COLORS["navy"], height=92)
        banner.pack(fill="both", expand=True)
        title_group = ttk.Frame(banner)
        title_group.configure(style="Header.TFrame")
        title_group.pack(side="left", padx=28, pady=16)
        ttk.Label(title_group, text=APP_TITLE, style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            title_group,
            text="輸入光學參數、執行多波長分析，並輸出 Zemax ZBF 場資料",
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        style = ttk.Style(self)
        style.configure("Header.TFrame", background=COLORS["navy"])

        outer = ttk.Frame(self, style="Page.TFrame")
        outer.pack(fill="both", expand=True, padx=18, pady=16)
        outer.columnconfigure(0, weight=0, minsize=460)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        self.settings_panel = ScrollableFrame(outer)
        self.settings_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self._build_settings(self.settings_panel.content)

        right = ttk.Frame(outer, style="Page.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_results(right)

        footer = ttk.Frame(self, style="Page.TFrame")
        footer.pack(fill="x", padx=20, pady=(0, 15))
        ttk.Label(
            footer,
            textvariable=self.vars["status"],
            foreground=COLORS["muted"],
            background=COLORS["bg"],
        ).pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=230)
        self.progress.pack(side="left", padx=16)
        self.stop_button = ttk.Button(
            footer, text="停止", style="Secondary.TButton",
            command=self.stop_analysis, state="disabled",
        )
        self.stop_button.pack(side="right", padx=(8, 0))
        self.run_button = ttk.Button(
            footer, text="開始分析並輸出", style="Primary.TButton",
            command=self.start_analysis,
        )
        self.run_button.pack(side="right")

    def _card(self, parent, title, subtitle=None):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.pack(fill="x", padx=(0, 8), pady=(0, 12))
        ttk.Label(card, text=title, style="Section.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(card, text=subtitle, style="Hint.TLabel", wraplength=390).pack(
                anchor="w", pady=(2, 11)
            )
        else:
            ttk.Separator(card).pack(fill="x", pady=(8, 11))
        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x")
        body.columnconfigure(1, weight=1)
        return body

    def _field(self, parent, row, label, variable, unit="", hint="", width=18):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=5
        )
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        if unit:
            ttk.Label(parent, text=unit, style="Field.TLabel").grid(
                row=row, column=2, sticky="w", padx=(7, 0), pady=5
            )
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").grid(
                row=row + 1, column=1, columnspan=2, sticky="w", pady=(0, 4)
            )
        return entry

    def _path_field(self, parent, row, label, variable, mode="directory"):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=5
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            parent,
            text="瀏覽",
            style="Secondary.TButton",
            command=lambda: self._browse_path(variable, mode),
        ).grid(row=row, column=2, padx=(7, 0), pady=5)

    def _build_settings(self, parent):
        basic = self._card(
            parent, "分析波長",
            "材料庫檔名會依波長自動對應為 asia_<nm>.npy；設計波長須包含於分析清單。",
        )
        self._field(basic, 0, "設計波長", self.vars["design_wavelength"], "nm")
        self._field(
            basic, 1, "分析波長", self.vars["wavelengths"], "nm",
            hint="以逗號分隔，例如：1290, 1310, 1330",
        )
        self._field(basic, 3, "最低強度穿透率", self.vars["min_transmission"])

        paths = self._card(parent, "檔案與環境")
        self._path_field(paths, 0, "Python", self.vars["python"], mode="file")
        self._path_field(paths, 1, "POP 光源資料夾", self.vars["source_dir"])
        self._path_field(paths, 2, "Meta-atom 資料庫", self.vars["library_dir"])
        self._path_field(paths, 3, "輸出資料夾", self.vars["output_dir"])

        geometry = self._card(parent, "光學與幾何參數")
        self._field(geometry, 0, "膠層折射率", self.vars["glue_index"])
        self._field(geometry, 1, "光源至 Metalens", self.vars["glue_distance"], "um")
        self._field(geometry, 2, "Metalens 直徑", self.vars["lens_diameter"], "um")
        self._field(geometry, 3, "Lens 後匯出位置", self.vars["after_lens"], "um")
        self._field(geometry, 4, "像素尺寸", self.vars["pixel_size"], "nm")
        self._field(geometry, 5, "模擬視窗", self.vars["field_size"], "um")

        advanced = self._card(
            parent, "進階計算",
            "較大的 padding 與掃描點數會增加記憶體用量及運算時間。",
        )
        ttk.Label(advanced, text="運算裝置", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=5
        )
        device = ttk.Combobox(
            advanced, textvariable=self.vars["device"],
            values=("auto", "cpu", "cuda"), state="readonly",
        )
        device.grid(row=0, column=1, sticky="ew", pady=5)
        self._field(advanced, 1, "Meta-atom alpha", self.vars["alpha"])
        self._field(advanced, 2, "Z 掃描範圍", self.vars["z_max"], "um")
        self._field(advanced, 3, "Beam sigma 點數", self.vars["z_points"])
        self._field(advanced, 4, "XZ 圖切片數", self.vars["xz_points"])
        self._field(advanced, 5, "Padding 網格點數", self.vars["pad_points"])

        exports = self._card(parent, "輸出內容")
        for row, (key, text) in enumerate((
            ("export_zbf", "ZBF（Zemax polarized beam）"),
            ("export_mat", "MAT（複數 Ex / Ey 場）"),
            ("export_plots", "PNG 圖表與 UI 預覽"),
        )):
            ttk.Checkbutton(exports, text=text, variable=self.vars[key]).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=4
            )

    def _build_results(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=0, column=0, sticky="nsew")
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        tools = ttk.Frame(card, style="Card.TFrame")
        tools.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(tools, text="執行與輸出", style="Section.TLabel").pack(side="left")
        ttk.Button(
            tools, text="開啟輸出資料夾", style="Secondary.TButton",
            command=self.open_output_folder,
        ).pack(side="right")
        ttk.Button(
            tools, text="清除紀錄", style="Secondary.TButton",
            command=self.clear_log,
        ).pack(side="right", padx=7)

        self.notebook = ttk.Notebook(card)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        log_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=8)
        summary_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=8)
        preview_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=8)
        files_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=8)
        self.notebook.add(log_tab, text="執行紀錄")
        self.notebook.add(summary_tab, text="結果摘要")
        self.notebook.add(preview_tab, text="圖表預覽")
        self.notebook.add(files_tab, text="輸出檔案")

        self.log_text = self._make_text(log_tab)
        self.summary_text = self._make_text(summary_tab)
        self.summary_text.insert("1.0", "分析完成後，這裡會顯示 wavelength_summary.txt。")
        self.summary_text.configure(state="disabled")

        preview_tab.rowconfigure(1, weight=1)
        preview_tab.columnconfigure(0, weight=1)
        chooser = ttk.Combobox(
            preview_tab, textvariable=self.vars["preview_choice"],
            state="readonly",
        )
        chooser.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        chooser.bind("<<ComboboxSelected>>", lambda _event: self.show_preview())
        self.preview_combo = chooser
        self.preview_label = ttk.Label(
            preview_tab,
            text="完成分析並輸出 PNG 後，可在此查看圖表。",
            anchor="center",
            background="#EEF2F7",
            foreground=COLORS["muted"],
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", lambda _event: self.show_preview())

        files_tab.rowconfigure(0, weight=1)
        files_tab.columnconfigure(0, weight=1)
        self.file_list = tk.Listbox(
            files_tab,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS["line"],
            activestyle="none",
            font=("Consolas", 10),
        )
        file_scroll = ttk.Scrollbar(files_tab, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scroll.set)
        self.file_list.grid(row=0, column=0, sticky="nsew")
        file_scroll.grid(row=0, column=1, sticky="ns")
        self.file_list.bind("<Double-Button-1>", self.open_selected_file)

    def _make_text(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        text = tk.Text(
            parent,
            wrap="none",
            borderwidth=0,
            background="#0E1726",
            foreground="#D9E5F4",
            insertbackground="white",
            font=("Cascadia Mono", 9),
            padx=12,
            pady=12,
        )
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        return text

    def _browse_path(self, variable, mode):
        current = Path(variable.get()).expanduser()
        if mode == "file":
            chosen = filedialog.askopenfilename(
                title="選擇 Python 執行檔",
                initialdir=str(current.parent if current.parent.exists() else REPO_ROOT),
                filetypes=(("Executable", "*.exe"), ("All files", "*.*")),
            )
        else:
            chosen = filedialog.askdirectory(
                title="選擇資料夾",
                initialdir=str(current if current.exists() else REPO_ROOT),
            )
        if chosen:
            variable.set(chosen)

    def _parsed_wavelengths(self):
        try:
            values = [
                float(item.strip())
                for item in self.vars["wavelengths"].get().split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise ValueError("分析波長必須是以逗號分隔的數字。") from exc
        if not values or any(value <= 0 for value in values):
            raise ValueError("請至少輸入一個大於 0 的分析波長。")
        return values

    @staticmethod
    def _nm_tag(value):
        return str(int(round(value))) if abs(value - round(value)) < 1e-8 else f"{value:g}"

    def _validate(self):
        python_path = Path(self.vars["python"].get().strip())
        if not python_path.is_file():
            raise ValueError("找不到指定的 Python 執行檔。")
        if not ANALYSIS_SCRIPT.is_file():
            raise ValueError(f"找不到分析腳本：{ANALYSIS_SCRIPT}")

        source_dir = Path(self.vars["source_dir"].get().strip())
        library_dir = Path(self.vars["library_dir"].get().strip())
        required_sources = [
            f"P3_10um_{pol}_{kind}.txt"
            for pol in ("EX", "EY")
            for kind in ("I", "Phase")
        ]
        missing_sources = [name for name in required_sources if not (source_dir / name).is_file()]
        if missing_sources:
            raise ValueError("POP 光源資料夾缺少：\n" + "\n".join(missing_sources))

        wavelengths = self._parsed_wavelengths()
        design = float(self.vars["design_wavelength"].get())
        if not any(abs(design - wavelength) < 1e-8 for wavelength in wavelengths):
            raise ValueError("設計波長必須包含於分析波長清單中。")
        missing_libraries = [
            f"asia_{self._nm_tag(wavelength)}.npy"
            for wavelength in wavelengths
            if not (library_dir / f"asia_{self._nm_tag(wavelength)}.npy").is_file()
        ]
        if missing_libraries:
            raise ValueError("Meta-atom 資料庫缺少：\n" + "\n".join(missing_libraries))

        numeric = {
            "最低穿透率": ("min_transmission", float),
            "膠層折射率": ("glue_index", float),
            "光源至 Metalens": ("glue_distance", float),
            "Metalens 直徑": ("lens_diameter", float),
            "Lens 後匯出位置": ("after_lens", float),
            "像素尺寸": ("pixel_size", float),
            "模擬視窗": ("field_size", float),
            "alpha": ("alpha", float),
            "Z 掃描範圍": ("z_max", float),
            "Beam sigma 點數": ("z_points", int),
            "XZ 圖切片數": ("xz_points", int),
            "Padding 網格點數": ("pad_points", int),
        }
        for label, (key, converter) in numeric.items():
            try:
                value = converter(self.vars[key].get())
            except ValueError as exc:
                raise ValueError(f"{label} 必須是有效數字。") from exc
            if key != "after_lens" and value <= 0:
                raise ValueError(f"{label} 必須大於 0。")
            if key == "after_lens" and value < 0:
                raise ValueError(f"{label} 不可小於 0。")
        transmission = float(self.vars["min_transmission"].get())
        if not 0 <= transmission <= 1:
            raise ValueError("最低穿透率必須介於 0 與 1 之間。")
        if not any(self.vars[key].get() for key in ("export_zbf", "export_mat", "export_plots")):
            raise ValueError("請至少選擇一種輸出內容。")

        output_dir = Path(self.vars["output_dir"].get().strip())
        output_dir.mkdir(parents=True, exist_ok=True)
        return wavelengths

    def _build_command(self, wavelengths):
        command = [
            self.vars["python"].get().strip(),
            "-u",
            str(ANALYSIS_SCRIPT),
            "--design-wavelength-nm", self.vars["design_wavelength"].get().strip(),
            "--wavelengths-nm", ",".join(self._nm_tag(value) for value in wavelengths),
            "--source-dir", self.vars["source_dir"].get().strip(),
            "--library-dir", self.vars["library_dir"].get().strip(),
            "--output-dir", self.vars["output_dir"].get().strip(),
            "--min-transmission", self.vars["min_transmission"].get().strip(),
            "--glue-index", self.vars["glue_index"].get().strip(),
            "--glue-distance-um", self.vars["glue_distance"].get().strip(),
            "--lens-diameter-um", self.vars["lens_diameter"].get().strip(),
            "--after-lens-um", self.vars["after_lens"].get().strip(),
            "--pixel-size-nm", self.vars["pixel_size"].get().strip(),
            "--field-size-um", self.vars["field_size"].get().strip(),
            "--alpha", self.vars["alpha"].get().strip(),
            "--device", self.vars["device"].get(),
            "--z-max-um", self.vars["z_max"].get().strip(),
            "--z-points", self.vars["z_points"].get().strip(),
            "--xz-points", self.vars["xz_points"].get().strip(),
            "--pad-points", self.vars["pad_points"].get().strip(),
        ]
        if not self.vars["export_zbf"].get():
            command.append("--no-zbf")
        if not self.vars["export_mat"].get():
            command.append("--no-mat")
        if not self.vars["export_plots"].get():
            command.append("--no-plots")
        return command

    def start_analysis(self):
        if self.process is not None:
            return
        try:
            wavelengths = self._validate()
            command = self._build_command(wavelengths)
        except (ValueError, OSError) as exc:
            messagebox.showerror("參數檢查失敗", str(exc), parent=self)
            return

        self._save_settings()
        self.clear_log()
        self._append_log("開始執行：\n" + subprocess.list2cmdline(command) + "\n\n")
        self.notebook.select(0)
        self.vars["status"].set("正在初始化分析環境…")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(12)

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._finish_run(False, f"無法啟動 Python：{exc}")
            return

        self.reader_thread = threading.Thread(
            target=self._read_process_output,
            args=(self.process,),
            daemon=True,
        )
        self.reader_thread.start()

    def _read_process_output(self, process):
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            self.output_queue.put(("line", line))
        return_code = process.wait()
        self.output_queue.put(("done", return_code))

    def _poll_process_output(self):
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "line":
                    line = str(payload)
                    self._append_log(line)
                    self._update_status_from_line(line)
                elif kind == "done":
                    self._finish_run(int(payload) == 0)
        except queue.Empty:
            pass
        self.after(100, self._poll_process_output)

    def _update_status_from_line(self, line):
        lower = line.lower()
        if "design @" in lower:
            self.vars["status"].set("已完成 Metalens 設計，正在套用寬頻材料限制…")
        elif "broadband transmission guard" in lower:
            self.vars["status"].set("正在執行各波長傳播與 ZBF 匯出…")
        elif "nm: ey rms" in lower:
            wavelength = line.split("nm:", 1)[0].strip()
            self.vars["status"].set(f"已完成 {wavelength} nm，繼續處理下一波長…")
        elif "all outputs saved" in lower:
            self.vars["status"].set("正在整理輸出結果…")

    def _finish_run(self, success, detail=""):
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.process = None
        if success:
            self.vars["status"].set("分析完成")
            self._append_log("\n✓ 分析完成。\n")
            self.refresh_results()
            self.notebook.select(1)
        else:
            self.vars["status"].set("分析失敗或已停止")
            if detail:
                self._append_log("\n" + detail + "\n")
            self._append_log("\n請查看上方錯誤訊息；常見原因是目前 Python 環境未安裝 PyTorch。\n")
            messagebox.showerror(
                "分析未完成",
                detail or "分析程序發生錯誤，請查看「執行紀錄」。",
                parent=self,
            )

    def stop_analysis(self):
        if self.process is None:
            return
        if not messagebox.askyesno("停止分析", "確定要停止目前的分析嗎？", parent=self):
            return
        self.vars["status"].set("正在停止…")
        try:
            self.process.terminate()
        except OSError:
            pass

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def _append_log(self, text):
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def refresh_results(self):
        output_dir = Path(self.vars["output_dir"].get().strip())
        summary_path = output_dir / "wavelength_summary.txt"
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        if summary_path.is_file():
            self.summary_text.insert("1.0", summary_path.read_text(encoding="utf-8", errors="replace"))
        else:
            self.summary_text.insert("1.0", "找不到 wavelength_summary.txt。")
        self.summary_text.configure(state="disabled")

        self.file_list.delete(0, "end")
        files = sorted(
            (path for path in output_dir.iterdir() if path.is_file()),
            key=lambda path: (path.suffix.lower(), path.name.lower()),
        ) if output_dir.is_dir() else []
        for path in files:
            self.file_list.insert("end", f"{path.name:<48} {self._format_size(path.stat().st_size):>10}")

        self.preview_paths = {
            path.stem: path
            for path in files
            if path.suffix.lower() in (".png", ".gif")
        }
        names = list(self.preview_paths)
        self.preview_combo.configure(values=names)
        if names:
            self.vars["preview_choice"].set(names[0])
            self.show_preview()
        else:
            self.vars["preview_choice"].set("")
            self.preview_image = None
            self.preview_label.configure(
                image="", text="此輸出資料夾沒有可預覽的 PNG 圖表。"
            )

    @staticmethod
    def _format_size(size):
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
            size /= 1024
        return f"{size:.1f} GB"

    def show_preview(self):
        name = self.vars["preview_choice"].get()
        path = self.preview_paths.get(name)
        if not path or not path.is_file():
            return
        try:
            image = tk.PhotoImage(file=str(path))
            available_w = max(1, self.preview_label.winfo_width() - 20)
            available_h = max(1, self.preview_label.winfo_height() - 20)
            factor = max(1, (image.width() + available_w - 1) // available_w,
                         (image.height() + available_h - 1) // available_h)
            if factor > 1:
                image = image.subsample(factor, factor)
            self.preview_image = image
            self.preview_label.configure(image=image, text="")
        except tk.TclError as exc:
            self.preview_image = None
            self.preview_label.configure(image="", text=f"無法預覽：{exc}")

    def open_output_folder(self):
        path = Path(self.vars["output_dir"].get().strip())
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def open_selected_file(self, _event=None):
        selection = self.file_list.curselection()
        if not selection:
            return
        name = self.file_list.get(selection[0]).split()[0]
        path = Path(self.vars["output_dir"].get().strip()) / name
        if path.exists():
            self._open_path(path)

    @staticmethod
    def _open_path(path):
        try:
            if os.name == "nt":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("無法開啟", str(exc))

    def _save_settings(self):
        data = {
            key: variable.get()
            for key, variable in self.vars.items()
            if key not in ("status", "preview_choice")
        }
        try:
            SETTINGS_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _load_settings(self):
        if not SETTINGS_FILE.is_file():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, value in data.items():
            if key in self.vars:
                self.vars[key].set(value)

    def on_close(self):
        if self.process is not None:
            if not messagebox.askyesno(
                "分析仍在執行",
                "關閉視窗會停止目前的分析，確定要關閉嗎？",
                parent=self,
            ):
                return
            try:
                self.process.terminate()
            except OSError:
                pass
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    MetalensAnalysisUI().mainloop()
