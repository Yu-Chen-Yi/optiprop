"""Verify an extracted portable ZIP, without relying on installed Python or Conda."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import struct
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import build_opener, ProxyHandler, Request
import zipfile

from build_support import forbidden_payload, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]


def isolated_environment(scratch):
    # In particular do not inherit PYTHON*, CONDA*, VIRTUAL_ENV, CUDA*, Qt, or PATH.
    permitted = {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PROCESSOR_ARCHITECTURE",
                 "NUMBER_OF_PROCESSORS", "PROGRAMDATA", "SYSTEMDRIVE"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in permitted}
    windows = Path(environment.get("SystemRoot", environment.get("SYSTEMROOT", r"C:\Windows")))
    environment["PATH"] = os.pathsep.join([str(windows / "System32"), str(windows)])
    for name in ("TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        target = scratch / name.lower()
        target.mkdir(parents=True, exist_ok=True)
        environment[name] = str(target)
    environment["MPLBACKEND"] = "Agg"
    environment["OPTIPROP_PACKAGING_LOG"] = str(scratch / "launcher.log")
    return environment


def assert_windowed_pe(path):
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise RuntimeError("Entry is not a Windows executable")
        stream.seek(0x3C)
        offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(offset)
        if stream.read(4) != b"PE\x00\x00":
            raise RuntimeError("Invalid PE header")
        machine = struct.unpack("<H", stream.read(2))[0]
        stream.seek(offset + 24 + 68)
        subsystem = struct.unpack("<H", stream.read(2))[0]
        if machine != 0x8664 or subsystem != 2:
            raise RuntimeError(f"Expected x64 windowed PE (machine={machine:#x}, subsystem={subsystem})")


def inspect_bundle(bundle):
    entry = bundle / "OptiProp.exe"
    assert_windowed_pe(entry)
    for required in ("_internal", "LICENSE", "README.md", "PORTABLE-README.md", "LICENSES",
                     "THIRD-PARTY-NOTICES.md", "dependency-manifest.json", "analysis-modules.json"):
        if not (bundle / required).exists():
            raise RuntimeError(f"Missing portable payload: {required}")
    if not list((bundle / "_internal").glob("python3*.dll")):
        raise RuntimeError("Embedded Python runtime DLL is missing")
    static = bundle / "_internal/optiprop/workbench/static"
    if not static.is_dir() or not any(static.rglob("*.*")):
        raise RuntimeError("Frozen static assets are missing")
    if not (list((bundle / "_internal/optiprop/workbench/examples").glob("*.json"))
            or list((bundle / "_internal/optiprop/examples").glob("*.json"))):
        raise RuntimeError("Frozen JSON examples are missing")
    paths = sorted(p for p in bundle.rglob("*") if p.is_file())
    bad = [p.relative_to(bundle).as_posix() for p in paths if forbidden_payload(p.relative_to(bundle))]
    if bad:
        raise RuntimeError("Forbidden Qt/GPU runtime in portable folder: " + ", ".join(bad))
    files = [{"path": p.relative_to(bundle).as_posix(), "bytes": p.stat().st_size,
              "sha256": sha256(p)} for p in paths if p.name != "bundle-manifest.json"]
    return {"entry": "OptiProp/OptiProp.exe", "file_count": len(files),
            "payload_bytes": sum(f["bytes"] for f in files), "files": files}


def extract_checked(archive, destination):
    with zipfile.ZipFile(archive) as zipped:
        seen = set()
        for member in zipped.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if (name.is_absolute() or ".." in name.parts or not name.parts
                    or name.parts[0] != "OptiProp" or any(":" in part for part in name.parts)):
                raise RuntimeError(f"Unsafe/non-portable ZIP member: {member.filename}")
            if member.filename.casefold() in seen:
                raise RuntimeError(f"Duplicate ZIP member: {member.filename}")
            seen.add(member.filename.casefold())
            if member.external_attr >> 16 & 0o170000 == 0o120000:
                raise RuntimeError("ZIP must not contain symbolic links")
            if not (destination / Path(*name.parts)).resolve().is_relative_to(destination.resolve()):
                raise RuntimeError("ZIP member escapes extraction directory")
        zipped.extractall(destination)


def validate_self_test(output):
    result = json.loads((output / "self-test.json").read_text(encoding="utf-8"))
    if result.get("status") != "passed" or "cuda_build" not in result or result["cuda_build"] is not None:
        raise RuntimeError("self-test.json must report passed with cuda_build=null")
    run = result.get("run", {})
    project = run.get("project", {})
    layers = project.get("optical_system", {}).get("layers", [])
    if (run.get("status") != "completed" or project.get("source", {}).get("type") != "IncidentSource"
            or not any(layer.get("type") == "Binary2LensLayer" for layer in layers)
            or not any(layer.get("type") == "PropagationLayer" and layer.get("spec", {}).get("method") == "asm"
                       for layer in layers)):
        raise RuntimeError("self-test.json must record a completed real source/Binary2/ASM run")
    return {"status": result["status"], "torch": result.get("torch"), "cuda_build": result["cuda_build"],
            "run_status": run["status"], "layers": [layer["type"] for layer in layers]}


def probe_http(entry, scratch, timeout):
    """Launch the exact frozen server, fetch assets/bootstrap, then request clean shutdown."""
    session_file = scratch / "http-session.json"
    environment = isolated_environment(scratch / "http-profile")
    command = [str(entry), "--no-browser", "--port", "0", "--output-dir", str(scratch / "http-output"),
               "--session-file", str(session_file)]
    session = None
    opener = build_opener(ProxyHandler({}))  # Loopback must not pass through host proxy settings.

    def request(path, data=None, authenticated=True):
        headers = {"Origin": session["origin"]}
        if authenticated:
            headers["X-OptiProp-Token"] = session["token"]
        if data is not None:
            headers["Content-Type"] = "application/json"
        with opener.open(Request(session["origin"] + path, data=data, headers=headers), timeout=10) as response:
            return response.read()

    with (scratch / "http-process.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=scratch / "empty working directory", env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + min(timeout, 120)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Frozen HTTP server exited before ready: {process.returncode}")
                if session_file.is_file():
                    try:
                        candidate = json.loads(session_file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        time.sleep(.1)
                        continue
                    origin = urlsplit(candidate["origin"])
                    if (origin.scheme != "http" or origin.hostname != "127.0.0.1" or not origin.port
                            or origin.username or origin.password or origin.path or origin.query or origin.fragment
                            or not candidate.get("token") or not candidate.get("url", "").startswith(candidate["origin"] + "/#")):
                        raise RuntimeError("Frozen session must identify an authenticated loopback-only server")
                    session = candidate
                    break
                time.sleep(.1)
            if session is None:
                raise RuntimeError("Timed out waiting for frozen HTTP session file")
            html = request("/", authenticated=False)
            javascript = request("/app.js", authenticated=False)
            if b"<html" not in html.lower() or not javascript:
                raise RuntimeError("Frozen HTTP static UI is missing or empty")
            bootstrap = json.loads(request("/api/bootstrap"))
            expected = {"free_propagation", "metalens_focus_ideal", "metalens_focus_binary2", "laser_collimator_demo"}
            if not expected.issubset(bootstrap.get("examples", {})):
                raise RuntimeError("Frozen bootstrap is missing expected packaged examples")
            try:
                request("/api/bootstrap", authenticated=False)
            except HTTPError as exc:
                if exc.code != 403:
                    raise
            else:
                raise RuntimeError("Frozen API accepted a missing session token")
            report = {"passed": True, "origin": session["origin"], "version": bootstrap.get("version"),
                      "html_bytes": len(html), "javascript_bytes": len(javascript),
                      "examples": sorted(bootstrap["examples"]), "missing_token_status": 403}
            request("/api/shutdown", data=b"{}")
            process.wait(timeout=30)
            if process.returncode != 0:
                raise RuntimeError(f"Frozen HTTP server shutdown failed: {process.returncode}")
            report["shutdown_returncode"] = process.returncode
            return report
        finally:
            # Only our own spawned process is controlled; never kill unrelated servers.
            if process.poll() is None:
                if session is not None:
                    try:
                        request("/api/shutdown", data=b"{}")
                        process.wait(timeout=10)
                    except Exception:
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)


def run_qa(archive, scratch_parent, timeout=300):
    archive = Path(archive).resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    # Relocation, spaces, Unicode, empty CWD and a scrubbed child environment are intentional.
    scratch = Path(tempfile.mkdtemp(prefix="OptiProp QA 測試-", dir=scratch_parent))
    report = {"archive": str(archive), "archive_sha256": sha256(archive),
              "archive_bytes": archive.stat().st_size, "scratch": str(scratch), "passed": False}
    try:
        extract_checked(archive, scratch)
        bundle = scratch / "OptiProp"
        actual = inspect_bundle(bundle)
        recorded = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
        if actual != recorded:
            raise RuntimeError("Extracted files do not match the recorded bundle manifest")
        report.update(entry=str(bundle / "OptiProp.exe"), payload_bytes=actual["payload_bytes"],
                      file_count=actual["file_count"])
        working = scratch / "empty working directory"
        working.mkdir()
        output = scratch / "smoke output"
        output.mkdir()
        command = [str(bundle / "OptiProp.exe"), "--self-test", "--no-browser", "--port", "0",
                   "--output-dir", str(output), "--session-file", str(scratch / "session.json")]
        report["command"] = command
        started = time.monotonic()
        with (scratch / "process.log").open("wb") as log:
            result = subprocess.run(command, cwd=working, env=isolated_environment(scratch),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        report.update(returncode=result.returncode, elapsed_seconds=round(time.monotonic() - started, 3))
        if result.returncode != 0:
            raise RuntimeError(f"Frozen --self-test failed ({result.returncode}); see {scratch}")
        report["self_test"] = validate_self_test(output)
        pngs = [p for p in output.rglob("*.png") if p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"]
        exports = [p for p in output.rglob("*") if p.suffix.lower() in {".npz", ".mat", ".zbf"}
                   and p.is_file() and p.stat().st_size > 0]
        if not pngs or {p.suffix.lower() for p in exports} != {".npz", ".mat", ".zbf"}:
            raise RuntimeError("Frozen self-test must produce a valid PNG plus NPZ, MAT and ZBF exports")
        report["smoke_files"] = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size,
                                  "sha256": sha256(p)} for p in sorted(pngs + exports)]
        report["http"] = probe_http(bundle / "OptiProp.exe", scratch, timeout)
        report["passed"] = True
        report["limitation"] = "Local isolation/relocation test, not a clean Windows VM certification."
        return report
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        write_json(scratch / "qa-report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Frozen Windows execution QA must run on Windows")
    report = run_qa(args.archive, ROOT / "tmp/packaging-qa", args.timeout)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
