"""Optional Qt desktop user interface for OptiProp.

Importing :mod:`optiprop` never imports Qt.  GUI symbols are loaded lazily so
the numerical package remains usable on servers without PySide6 installed.
"""

from __future__ import annotations

from typing import Any


def launch(argv: list[str] | None = None) -> int:
    """Start the desktop application and return its process exit code."""

    from .app import main

    return main(argv)


def __getattr__(name: str) -> Any:
    if name == "MainWindow":
        from .main_window import MainWindow

        return MainWindow
    raise AttributeError(name)


__all__ = ["MainWindow", "launch"]
