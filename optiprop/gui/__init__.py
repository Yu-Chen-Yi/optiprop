"""Compatibility launcher for OptiProp's local browser workbench.

The former Qt widgets are no longer part of this package. Importing this
module does not start a server or open a browser.
"""

from __future__ import annotations


def launch(argv: list[str] | None = None) -> int:
    """Start the browser workbench and return its process exit code."""

    from .app import main

    return main(argv)


__all__ = ["launch"]
