"""Pure helpers and strict environment guards shared by the spec and build CLI."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys

EXCLUDED_MODULES = [
    "PySide", "PySide2", "PySide6", "PyQt4", "PyQt5", "PyQt6", "shiboken2", "shiboken6",
    "tkinter", "Tkinter", "_tkinter", "Tk", "tcl", "PIL.ImageQt", "PIL.ImageTk",
    "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qtcairo", "matplotlib.backends.backend_qt5",
    "matplotlib.backends.backend_qt5agg", "matplotlib.backends.backend_qt5cairo",
    "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_tkcairo",
    "matplotlib.backends._backend_tk",
    "nvidia", "cuda", "cupy", "cupyx", "pycuda", "triton", "pytorch_triton",
    "tensorrt", "torch_tensorrt", "tensorflow", "jax", "jaxlib", "qtpy", "qtconsole",
    "torchvision", "torchaudio", "optiprop.gui",
]
# torch.cuda Python stubs are deliberately retained: CPU torch imports them too.
_GPU_DLL = re.compile(
    r"^(?:torch_cuda|c10_cuda|cudart|cudnn|cublas|cufft|curand|cusolver|cusparse|"
    r"nvrtc|nvjitlink|nvcuda|nccl|amdhip|hipblas|rocblas).*\.(?:dll|so|dylib)(?:\..*)?$",
    re.I,
)


def normalized_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def forbidden_distributions(names):
    rejected = []
    for name in names:
        value = normalized_name(name)
        if (value.startswith(("pyside", "pyqt", "shiboken", "nvidia-", "cupy", "cuda-", "pytorch-triton"))
                or value in {"triton", "pycuda", "tensorrt", "torch-tensorrt", "qtpy", "qtconsole"}):
            rejected.append(name)
    return sorted(rejected)


def forbidden_module(name):
    return any(name == excluded or name.startswith(excluded + ".") for excluded in EXCLUDED_MODULES)


def scipy_compat_hiddenimports(scipy_directory):
    # Recent SciPy moved this vendored namespace from _lib to _external. Its numpy
    # shim imports FFT/linalg via __import__(__package__ + ...), which static
    # analysis misses; PyInstaller 6.20's upstream hook still names the old path.
    package = Path(scipy_directory) / "_external/array_api_compat/numpy"
    if package.is_dir():
        return ["scipy._external.array_api_compat.numpy.fft",
                "scipy._external.array_api_compat.numpy.linalg"]
    return []


def forbidden_payload(name):
    parts = PurePosixPath(str(name).replace("\\", "/")).parts
    for part in parts:
        lower = part.lower()
        if lower.startswith(("pyside", "pyqt", "shiboken")):
            return True
        if lower in {"tkinter", "_tkinter.pyd", "tcl", "tk", "nvidia", "triton", "cupy"}:
            return True
        if re.match(r"^(?:qt[456].*|tcl\d.*|tk\d.*)\.dll$", lower):
            return True
        if _GPU_DLL.match(part):
            return True
    return False


def assert_build_environment():
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Build the Windows x64 executable on Windows x64, not by cross-compiling")
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve():
        raise RuntimeError("Use the isolated tmp/build-venv Python, never the base/Conda interpreter")
    config = prefix / "pyvenv.cfg"
    if not config.is_file() or not re.search(
        r"^include-system-site-packages\s*=\s*false\s*$", config.read_text(), re.M | re.I
    ):
        raise RuntimeError("Build venv must disable system site packages")
    distributions = list(metadata.distributions())
    rejected = forbidden_distributions(d.metadata.get("Name", "") for d in distributions)
    if rejected:
        raise RuntimeError("Qt/GPU distributions installed in build venv: " + ", ".join(rejected))
    for name in ("torch", "numpy", "scipy", "matplotlib", "scikit-image", "rich", "pyinstaller"):
        dist = metadata.distribution(name)  # Missing dependencies fail before analysis.
        if not Path(dist.locate_file("")).resolve().is_relative_to(prefix):
            raise RuntimeError(f"{name} resolves outside isolated venv")
    import torch
    if not Path(torch.__file__).resolve().is_relative_to(prefix):
        raise RuntimeError("torch was imported from outside isolated venv")
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("Portable build requires CPU torch: torch.version.cuda and hip must be None")
    # Inspect native files as well; do not accept an incorrectly labeled wheel.
    bad = [str(p.relative_to(prefix)) for p in Path(torch.__file__).parent.rglob("*.dll")
           if forbidden_payload(p.name)]
    if bad:
        raise RuntimeError("GPU runtime found in torch wheel: " + ", ".join(bad))
    return {"python": platform.python_version(), "architecture": platform.machine(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "torch_hip": getattr(torch.version, "hip", None),
            "pyinstaller": metadata.version("pyinstaller"), "build_python": sys.executable}


def read_version(root):
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    version = re.search(r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"\s*$',
                        section.group(1) if section else "", re.M)
    if not version:
        raise RuntimeError("Expected a static X.Y.Z project version in pyproject.toml")
    return version.group(1)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_notices(root, stage, environment):
    """Preserve installed wheels' notice files and declare full build provenance."""
    licenses = stage / "LICENSES"
    licenses.mkdir()
    dependencies = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata.get("Name", "").lower()):
        name = dist.metadata.get("Name", "unknown")
        notices = []
        for item in dist.files or []:
            relative = PurePosixPath(str(item).replace("\\", "/"))
            if ".." in relative.parts or relative.is_absolute():
                continue
            basename = relative.name.lower()
            if not (basename.startswith(("license", "licence", "copying", "notice", "copyright"))
                    or "licenses" in (p.lower() for p in relative.parts)):
                continue
            source = Path(dist.locate_file(item))
            if not source.is_file() or source.suffix.lower() in {".py", ".pyc", ".pyd", ".dll"}:
                continue
            target = licenses / normalized_name(name) / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            notices.append(target.relative_to(stage).as_posix())
        dependencies.append({"name": name, "version": dist.version,
                             "license_expression": dist.metadata.get("License-Expression"),
                             "license_metadata": dist.metadata.get("License"),
                             "license_classifiers": [c for c in dist.metadata.get_all("Classifier", [])
                                                     if c.startswith("License ::")],
                             "notice_files": notices})
    for candidate in ("LICENSE.txt", "LICENSE_PYTHON.txt", "LICENSE"):
        source = Path(sys.base_prefix) / candidate
        if source.is_file():
            shutil.copy2(source, licenses / "PYTHON-LICENSE.txt")
            break
    else:
        raise RuntimeError("Python runtime license not found in base interpreter directory")
    shutil.copy2(root / "LICENSE", stage / "LICENSE")
    shutil.copy2(root / "README.md", stage / "README.md")
    shutil.copy2(root / "packaging/windows/PORTABLE-README.md", stage / "PORTABLE-README.md")
    write_json(stage / "dependency-manifest.json", {
        "schema_version": 1, "optiprop_version": read_version(root),
        "scope": "Complete build-environment inventory; build tools may not be bundled. "
                 "analysis-modules.json lists frozen Python modules; bundle-manifest.json lists shipped files.",
        "environment": environment, "dependencies": dependencies,
    })
    lines = ["# Third-party notices", "", "OptiProp is MIT licensed; see LICENSE.", "",
             "The portable bundle contains Python and third-party software under their own licenses.",
             "Original wheel notices (including nested native-library notices) are retained in LICENSES/.",
             "Python's license is LICENSES/PYTHON-LICENSE.txt. This inventory also includes build tools;",
             "consult analysis-modules.json and bundle-manifest.json for actual shipped modules/files.", "",
             "The absence of a notice file in a wheel does not imply public-domain software.", ""]
    for dependency in dependencies:
        count = len(dependency["notice_files"])
        lines.append(f"- {dependency['name']} {dependency['version']}: {count} notice file(s); "
                     "license metadata is recorded in dependency-manifest.json.")
    (stage / "THIRD-PARTY-NOTICES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
