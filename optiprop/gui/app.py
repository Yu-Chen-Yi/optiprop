"""Command-line application entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    parser = argparse.ArgumentParser(prog="optiprop-gui", description="OptiProp near-field multilayer workbench")
    parser.add_argument("project", nargs="?", help="Project JSON/OPROP file to open")
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(["optiprop-gui", *qt_args])
    QCoreApplication.setOrganizationName("OptiProp")
    QCoreApplication.setApplicationName("NearFieldWorkbench")
    window = MainWindow()
    if args.project:
        from pathlib import Path
        from optiprop.project import load_project, require_valid_assets

        path = Path(args.project).expanduser().resolve()
        project = load_project(path)
        require_valid_assets(project.assets, path.parent)
        window.load_dict(project.to_dict(), base_dir=path.parent)
        window.project_path = path
        window.setWindowTitle(f"{path.name} — OptiProp")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
