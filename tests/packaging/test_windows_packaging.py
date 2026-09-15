"""Stdlib-only tests: safe to run while the isolated build venv is provisioning."""

from __future__ import annotations

import ast
from email.message import Message
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TOOLING = ROOT / "packaging/windows"
sys.path.insert(0, str(TOOLING))
import build_support as support
import qa


def write_pe(path, subsystem=2, machine=0x8664):
    data = bytearray(256)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\x00\x00"
    struct.pack_into("<H", data, 0x84, machine)
    struct.pack_into("<H", data, 0x80 + 24 + 68, subsystem)
    path.write_bytes(data)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "tmp/packaging-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_forbidden_distribution_names(self):
        bad = ["PySide6", "PySide6_Addons", "PyQt6-Qt6", "shiboken6", "nvidia-cublas-cu12",
               "cupy-cuda12x", "triton", "cuda-python", "qtpy", "qtconsole", "pytorch-triton-rocm"]
        self.assertEqual(support.forbidden_distributions(bad + ["torch", "numpy", "rich"]), sorted(bad))

    def test_exclusions_keep_cpu_torch_compatibility_stubs(self):
        for module in ["PySide6.QtCore", "PyQt5", "shiboken6", "tkinter.filedialog", "Tk",
                       "PIL.ImageQt", "PIL.ImageTk", "nvidia.cublas", "cupy", "optiprop.gui"]:
            self.assertTrue(support.forbidden_module(module), module)
        for module in ["torch", "torch.cuda", "numpy", "PIL.Image", "optiprop.workbench.app"]:
            self.assertFalse(support.forbidden_module(module), module)

    def test_native_payload_guard(self):
        for name in ["_internal/PySide6/QtCore.pyd", r"_internal\Qt6Core.dll", "torch/lib/torch_cuda.dll",
                     "torch/lib/c10_cuda.dll", "cudnn64_9.dll", "cublas64_12.dll", "nvrtc64_120_0.dll",
                     "_internal/_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "nvidia/foo.dat"]:
            self.assertTrue(support.forbidden_payload(name), name)
        for name in ["torch/lib/torch_cpu.dll", "torch/lib/c10.dll", "torch/cuda/__init__.py",
                     "python312.dll", "_internal/optiprop/workbench/static/app.js"]:
            self.assertFalse(support.forbidden_payload(name), name)

    def test_reject_base_interpreter_before_importing_torch(self):
        with mock.patch.object(support.sys, "platform", "win32"), \
             mock.patch.object(support.platform, "machine", return_value="AMD64"), \
             mock.patch.object(support.sys, "prefix", str(self.directory)), \
             mock.patch.object(support.sys, "base_prefix", str(self.directory)):
            with self.assertRaisesRegex(RuntimeError, "base/Conda"):
                support.assert_build_environment()

    def environment_patches(self, cuda=None, hip=None, installed=()):
        venv = self.directory / "venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
        fake_dist = types.SimpleNamespace(locate_file=lambda _: venv)
        fake_torch = types.SimpleNamespace(__file__=str(venv / "torch/__init__.py"),
                                           __version__="2.5.1+cpu",
                                           version=types.SimpleNamespace(cuda=cuda, hip=hip))
        stack = __import__("contextlib").ExitStack()
        self.addCleanup(stack.close)
        for target, attribute, value in [(support.sys, "platform", "win32"),
                                          (support.sys, "prefix", str(venv)),
                                          (support.sys, "base_prefix", str(self.directory))]:
            stack.enter_context(mock.patch.object(target, attribute, value))
        stack.enter_context(mock.patch.object(support.platform, "machine", return_value="AMD64"))
        stack.enter_context(mock.patch.object(support.metadata, "distributions", return_value=[
            types.SimpleNamespace(metadata={"Name": name}) for name in installed]))
        stack.enter_context(mock.patch.object(support.metadata, "distribution", return_value=fake_dist))
        stack.enter_context(mock.patch.object(support.metadata, "version", return_value="6.20.0"))
        stack.enter_context(mock.patch.dict(sys.modules, {"torch": fake_torch}))
        return venv

    def test_installed_qt_rejected_even_if_unused(self):
        self.environment_patches(installed=["PySide6"])
        with self.assertRaisesRegex(RuntimeError, "Qt/GPU distributions"):
            support.assert_build_environment()

    def test_cuda_wheel_rejected_without_checking_hardware(self):
        self.environment_patches(cuda="12.4")
        with self.assertRaisesRegex(RuntimeError, "requires CPU torch"):
            support.assert_build_environment()

    def test_rocm_wheel_rejected(self):
        self.environment_patches(hip="6.2")
        with self.assertRaisesRegex(RuntimeError, "requires CPU torch"):
            support.assert_build_environment()

    def test_cpu_venv_accepted(self):
        self.environment_patches()
        result = support.assert_build_environment()
        self.assertIsNone(result["torch_cuda"])
        self.assertEqual(result["torch"], "2.5.1+cpu")

    def test_system_site_packages_rejected(self):
        venv = self.environment_patches()
        (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
        with self.assertRaisesRegex(RuntimeError, "disable system site"):
            support.assert_build_environment()

    def test_dependency_outside_venv_rejected(self):
        self.environment_patches()
        with mock.patch.object(support.metadata, "distribution", return_value=types.SimpleNamespace(
            locate_file=lambda _: self.directory
        )):
            with self.assertRaisesRegex(RuntimeError, "outside isolated venv"):
                support.assert_build_environment()

    def test_child_environment_has_no_developer_runtime(self):
        with mock.patch.dict(os.environ, {"PATH": r"C:\conda;C:\CUDA", "PYTHONPATH": "secret",
                                         "CONDA_PREFIX": "secret", "SystemRoot": r"C:\Windows"}, clear=True):
            environment = qa.isolated_environment(self.directory)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("CONDA_PREFIX", environment)
        self.assertNotIn("conda", environment["PATH"])
        self.assertNotIn("CUDA", environment["PATH"])
        self.assertEqual(environment["MPLBACKEND"], "Agg")
        self.assertEqual(environment["USERPROFILE"], str(self.directory / "userprofile"))

    def test_pe_windowed_x64_contract(self):
        exe = self.directory / "OptiProp.exe"
        write_pe(exe)
        qa.assert_windowed_pe(exe)
        write_pe(exe, subsystem=3)
        with self.assertRaisesRegex(RuntimeError, "windowed"):
            qa.assert_windowed_pe(exe)
        write_pe(exe, machine=0x14C)
        with self.assertRaisesRegex(RuntimeError, "x64"):
            qa.assert_windowed_pe(exe)

    def test_zip_rejects_path_traversal_and_absolute_paths(self):
        for index, name in enumerate(["../escape.txt", "OptiProp/../../escape.txt", "/OptiProp/file",
                                      "OptiProp/C:/escape", "Other/file"]):
            archive = self.directory / f"{index}.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr(name, "bad")
            with self.assertRaisesRegex(RuntimeError, "Unsafe/non-portable"):
                qa.extract_checked(archive, self.directory / "extract")

    def test_zip_preserves_portable_root(self):
        archive = self.directory / "valid.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("OptiProp/_internal/file", "runtime")
        qa.extract_checked(archive, self.directory / "extract")
        self.assertEqual((self.directory / "extract/OptiProp/_internal/file").read_text(), "runtime")

    def test_failed_qa_always_leaves_report(self):
        archive = self.directory / "broken.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("OptiProp/OptiProp.exe", "not a PE")
        with self.assertRaisesRegex(RuntimeError, "not a Windows executable"):
            qa.run_qa(archive, self.directory / "qa")
        report_path = next((self.directory / "qa").rglob("qa-report.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])
        self.assertIn("not a Windows executable", report["error"])

    def self_test_document(self):
        return {"status": "passed", "torch": "2.5.1+cpu", "cuda_build": None, "run": {
            "status": "completed", "project": {"source": {"type": "IncidentSource"},
            "optical_system": {"layers": [{"type": "Binary2LensLayer"},
                                           {"type": "PropagationLayer", "spec": {"method": "asm"}}]}}}}

    def test_self_test_requires_cpu_and_actual_optical_project(self):
        result = self.self_test_document()
        report = self.directory / "self-test.json"
        report.write_text(json.dumps(result))
        self.assertEqual(qa.validate_self_test(self.directory)["status"], "passed")
        result["cuda_build"] = "12.4"
        report.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, "cuda_build=null"):
            qa.validate_self_test(self.directory)
        result["cuda_build"] = None
        result["run"]["project"]["optical_system"]["layers"] = []
        report.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, "real source/Binary2/ASM"):
            qa.validate_self_test(self.directory)

    def test_version_is_read_without_importing_project(self):
        (self.directory / "pyproject.toml").write_text('[project]\nname = "optiprop"\nversion = "1.0.8"\n')
        self.assertEqual(support.read_version(self.directory), "1.0.8")

    def test_all_packaging_python_and_spec_parse(self):
        for path in [*TOOLING.glob("*.py"), *TOOLING.glob("*.spec")]:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_notices_keep_nested_wheel_licenses_and_python_license(self):
        wheel_root = self.directory / "site-packages"
        relative = Path("example-1.dist-info/licenses/native/LICENSE.txt")
        source = wheel_root / relative
        source.parent.mkdir(parents=True)
        source.write_text("Native library license")
        metadata = Message()
        metadata["Name"] = "example"
        metadata["License-Expression"] = "MIT"
        dist = types.SimpleNamespace(metadata=metadata, version="1.0", files=[relative],
                                     locate_file=lambda path: wheel_root / path)
        stage = self.directory / "stage"
        stage.mkdir()
        (self.directory / "LICENSE_PYTHON.txt").write_text("Python runtime license")
        with mock.patch.object(support.metadata, "distributions", return_value=[dist]), \
             mock.patch.object(support.sys, "base_prefix", str(self.directory)):
            support.write_notices(ROOT, stage, {"torch": "2.5.1+cpu", "torch_cuda": None})
        manifest = json.loads((stage / "dependency-manifest.json").read_text(encoding="utf-8"))
        dependency = manifest["dependencies"][0]
        self.assertEqual(dependency["license_expression"], "MIT")
        self.assertEqual(len(dependency["notice_files"]), 1)
        self.assertEqual((stage / dependency["notice_files"][0]).read_text(), "Native library license")
        self.assertEqual((stage / "LICENSES/PYTHON-LICENSE.txt").read_text(), "Python runtime license")


if __name__ == "__main__":
    unittest.main()
