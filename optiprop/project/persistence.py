"""Strict JSON loading, migration, and crash-safe atomic saving."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import ProjectFormatError, UnsupportedProjectVersionError
from .models import CURRENT_SCHEMA_VERSION, PROJECT_FORMAT, OptiPropProject


def save_project(project: OptiPropProject, path: str | Path) -> Path:
    """Atomically replace ``path`` with deterministic UTF-8 project JSON."""

    if not isinstance(project, OptiPropProject):
        raise TypeError("project must be an OptiPropProject.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        project.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_project(path: str | Path) -> OptiPropProject:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectFormatError(f"Could not read project {source}: {exc}") from exc
    migrated = migrate_project_document(document)
    try:
        return OptiPropProject.from_dict(migrated)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectFormatError(f"Invalid project document: {exc}") from exc


def migrate_project_document(document: object) -> dict[str, Any]:
    """Normalize legacy v1 spellings and reject unknown schema versions."""

    if not isinstance(document, dict):
        raise ProjectFormatError("Project document root must be an object.")
    migrated = dict(document)
    if "schema_version" not in migrated:
        # Early WP-10 drafts used `version`, `id`, and `system`.  Treat only
        # an explicit version 1 document as migratable, never guess arbitrary JSON.
        if migrated.get("version") != 1:
            raise ProjectFormatError("Project schema_version is missing.")
        migrated["schema_version"] = migrated.pop("version")
        if "id" in migrated and "project_id" not in migrated:
            migrated["project_id"] = migrated.pop("id")
        if "system" in migrated and "optical_system" not in migrated:
            migrated["optical_system"] = migrated.pop("system")
        migrated.setdefault("format", PROJECT_FORMAT)
    version = migrated.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProjectFormatError("schema_version must be an integer.")
    if version != CURRENT_SCHEMA_VERSION:
        raise UnsupportedProjectVersionError(
            f"Unsupported project schema version {version}; "
            f"this build supports version {CURRENT_SCHEMA_VERSION}."
        )
    if migrated.get("format") != PROJECT_FORMAT:
        raise ProjectFormatError(f"format must be {PROJECT_FORMAT!r}.")
    return migrated


__all__ = ["load_project", "migrate_project_document", "save_project"]
