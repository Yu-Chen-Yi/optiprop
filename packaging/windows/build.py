"""Build a CPU-only portable folder, ZIP it, and verify that exact ZIP before handoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

from build_support import assert_build_environment, read_version, sha256, write_json, write_notices
from qa import inspect_bundle, run_qa

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true", help="Read-only environment check; do not build")
    parser.add_argument("--qa-timeout", type=int, default=300)
    args = parser.parse_args()
    environment = assert_build_environment()
    print(json.dumps(environment, indent=2), flush=True)
    if args.check_env:
        return 0
    if not (ROOT / "optiprop/workbench/app.py").is_file():
        raise RuntimeError("Workbench app.py is not ready; wait for the application owner")
    version = read_version(ROOT)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    scratch = ROOT / "tmp/packaging-build" / run
    destination = ROOT / "dist/windows" / run
    scratch.mkdir(parents=True, exist_ok=False)
    destination.mkdir(parents=True, exist_ok=False)
    stage = scratch / "metadata"
    stage.mkdir()
    write_notices(ROOT, stage, environment)
    source_paths = [ROOT / "pyproject.toml", ROOT / "LICENSE", ROOT / "README.md"]
    source_paths.extend(p for p in (ROOT / "optiprop").rglob("*") if p.is_file()
                        and p.suffix.lower() in {".py", ".json", ".js", ".css", ".html", ".svg"}
                        and "__pycache__" not in p.parts)
    write_json(stage / "source-manifest.json", [
        {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in sorted(source_paths)
    ])
    build_env = os.environ.copy()
    for key in list(build_env):
        if key.upper().startswith(("PYTHON", "CONDA", "QT_")) or key.upper() == "VIRTUAL_ENV":
            del build_env[key]
    # Preserve only the chosen venv plus Windows tools in subprocess DLL/PATH search.
    windows = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    build_env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent),
                                       str(windows / "System32"), str(windows)])
    build_env["PYTHONNOUSERSITE"] = "1"
    build_env["MPLBACKEND"] = "Agg"
    build_env["PYINSTALLER_CONFIG_DIR"] = str(scratch / "pyinstaller-cache")
    build_env["OPTIPROP_BUILD_METADATA"] = str(stage)
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
               "--distpath", str(destination), "--workpath", str(scratch / "analysis"),
               str(ROOT / "packaging/windows/OptiProp.spec")]
    print(f"Build log: {scratch / 'pyinstaller.log'}", flush=True)
    with (scratch / "pyinstaller.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=build_env, stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"PyInstaller failed ({process.returncode}); see {scratch / 'pyinstaller.log'}")
    bundle = destination / "OptiProp"
    shutil.copytree(stage, bundle, dirs_exist_ok=True)
    write_json(bundle / "bundle-manifest.json", inspect_bundle(bundle))
    archive = destination / f"OptiProp-{version}-windows-x64.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(destination).as_posix())
    print(f"Verifying exact ZIP: {archive}", flush=True)
    qa = run_qa(archive, ROOT / "tmp/packaging-qa", args.qa_timeout)
    digest = sha256(archive)
    (destination / (archive.name + ".sha256")).write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    report = {"verified": True, "version": version, "archive": str(archive),
              "archive_bytes": archive.stat().st_size, "sha256": digest,
              "entry": str(bundle / "OptiProp.exe"), "portable_folder": str(bundle),
              "build_log": str(scratch / "pyinstaller.log"), "environment": environment, "qa": qa}
    write_json(destination / "build-report.json", report)
    write_json(ROOT / "tmp/packaging-latest.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
