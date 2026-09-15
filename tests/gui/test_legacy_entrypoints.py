"""Qt-free compatibility tests; the browser backend has its own test suite."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from optiprop.gui import launch
from optiprop.gui.app import main


@pytest.mark.parametrize("entrypoint", [main, launch])
@pytest.mark.parametrize("argv", [None, [], ["example.json"], ["--help"]])
def test_legacy_launchers_forward_arguments_and_exit_code(monkeypatch, entrypoint, argv):
    backend = ModuleType("optiprop.workbench.app")
    backend.main = Mock(return_value=17)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    assert entrypoint(argv) == 17
    backend.main.assert_called_once_with(argv)


def test_legacy_module_entrypoint_preserves_system_exit(monkeypatch):
    backend = ModuleType("optiprop.workbench.app")
    backend.main = Mock(return_value=23)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "optiprop/gui/app.py"),
            run_name="__main__",
        )

    assert exc.value.code == 23
    backend.main.assert_called_once_with(None)


def test_legacy_imports_are_lazy_and_do_not_import_qt():
    code = """
import importlib.abc
import sys

blocked = {"PySide2", "PySide6", "PyQt5", "PyQt6", "qtpy", "shiboken2", "shiboken6"}

class NoQtOrWorkbench(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked or fullname.startswith("optiprop.workbench"):
            raise AssertionError(f"Unexpected eager import: {fullname}")

sys.meta_path.insert(0, NoQtOrWorkbench())
import optiprop
import optiprop.gui
import optiprop.gui.app
assert optiprop.gui.__all__ == ["launch"]
assert not hasattr(optiprop.gui, "MainWindow")
assert not any(name.split(".")[0] in blocked for name in sys.modules)
assert not any(name.startswith("optiprop.workbench") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
