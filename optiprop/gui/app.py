"""Compatibility entry point for the local browser workbench (no Qt)."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Forward arguments unchanged; legacy Qt project files are not migrated."""
    from optiprop.workbench.app import main as workbench_main

    return workbench_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
