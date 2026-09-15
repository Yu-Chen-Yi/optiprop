# Build only through packaging/windows/build.py (supplies a fresh metadata stage).
from pathlib import Path
import os
import sys

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent.parent
sys.path.insert(0, str(SPEC_DIR))
sys.path.insert(0, str(ROOT))

from build_support import assert_build_environment, EXCLUDED_MODULES, forbidden_module, forbidden_payload
from PyInstaller.utils.hooks import collect_data_files

assert_build_environment()
stage = Path(os.environ["OPTIPROP_BUILD_METADATA"]).resolve()
os.environ["MPLBACKEND"] = "Agg"

# An allowlist, never a Tree(ROOT): no user fields, results, experiments or source trees.
datas = collect_data_files(
    "optiprop",
    includes=["workbench/static/*", "workbench/static/**/*",
              "workbench/examples/*.json", "examples/*.json"],
    excludes=["**/__pycache__/**"],
)
if not any("static" in Path(src).parts for src, _ in datas):
    raise RuntimeError("Workbench static assets are missing")
if not any(Path(src).suffix == ".json" for src, _ in datas):
    raise RuntimeError("Packaged JSON examples are missing")

a = Analysis(
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["matplotlib.backends.backend_agg"],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[str(SPEC_DIR / "runtime_hook.py")],
    excludes=EXCLUDED_MODULES,
    noarchive=False,
)

# Fail closed if a hook reintroduces a forbidden toolkit or accelerator runtime.
bad = [name for name, *_ in a.pure if forbidden_module(name)]
bad += [name for name, *_ in a.binaries + a.datas if forbidden_payload(name)]
if bad:
    raise RuntimeError("Forbidden modules/runtime files in Analysis: " + ", ".join(bad))

import json
(stage / "analysis-modules.json").write_text(
    json.dumps(sorted(name for name, *_ in a.pure), indent=2) + "\n", encoding="utf-8"
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OptiProp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="OptiProp")
