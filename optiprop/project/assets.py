"""Portable, content-addressed project asset references."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Iterable
from uuid import UUID, uuid4

from .errors import AssetIntegrityError, UnsafeAssetPathError


class AssetStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    CHANGED = "changed"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class AssetReference:
    """A relative file reference pinned to its SHA-256 content digest."""

    path: str
    sha256: str
    size_bytes: int
    kind: str = "field"
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id))
        object.__setattr__(self, "path", normalize_asset_path(self.path))
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be a 64-character hexadecimal digest.")
        object.__setattr__(self, "sha256", digest)
        if isinstance(self.size_bytes, bool) or int(self.size_bytes) < 0:
            raise ValueError("size_bytes must be a non-negative integer.")
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("kind must not be empty.")
        object.__setattr__(self, "kind", kind)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        project_directory: str | Path,
        kind: str = "field",
        id: UUID | str | None = None,
    ) -> "AssetReference":
        base = Path(project_directory).resolve()
        target = Path(path).resolve()
        try:
            relative = target.relative_to(base)
        except ValueError as exc:
            raise UnsafeAssetPathError(
                "Project assets must be inside the project directory."
            ) from exc
        if not target.is_file():
            raise FileNotFoundError(target)
        return cls(
            id=uuid4() if id is None else _uuid(id),
            path=relative.as_posix(),
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
            kind=kind,
        )

    def resolve(self, project_directory: str | Path) -> Path:
        base = Path(project_directory).resolve()
        target = (base / Path(*PurePosixPath(self.path).parts)).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise UnsafeAssetPathError(
                f"Asset path escapes the project directory: {self.path!r}."
            ) from exc
        return target

    def to_dict(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, value: object) -> "AssetReference":
        if not isinstance(value, dict):
            raise TypeError("Asset reference must be an object.")
        try:
            return cls(
                id=value["id"],
                path=value["path"],
                sha256=value["sha256"],
                size_bytes=value["size_bytes"],
                kind=value.get("kind", "field"),
            )
        except KeyError as exc:
            raise ValueError(f"Missing asset property {exc.args[0]!r}.") from exc


@dataclass(frozen=True)
class AssetCheck:
    asset: AssetReference
    status: AssetStatus
    resolved_path: Path | None = None
    actual_sha256: str | None = None
    message: str = ""


def validate_assets(
    assets: Iterable[AssetReference], project_directory: str | Path
) -> tuple[AssetCheck, ...]:
    checks: list[AssetCheck] = []
    for asset in assets:
        try:
            target = asset.resolve(project_directory)
        except UnsafeAssetPathError as exc:
            checks.append(AssetCheck(asset, AssetStatus.UNSAFE, message=str(exc)))
            continue
        if not target.is_file():
            checks.append(
                AssetCheck(asset, AssetStatus.MISSING, target, message="Asset is missing.")
            )
            continue
        actual = sha256_file(target)
        if actual != asset.sha256 or target.stat().st_size != asset.size_bytes:
            checks.append(
                AssetCheck(
                    asset,
                    AssetStatus.CHANGED,
                    target,
                    actual,
                    "Asset content differs from the saved project.",
                )
            )
        else:
            checks.append(AssetCheck(asset, AssetStatus.OK, target, actual))
    return tuple(checks)


def require_valid_assets(
    assets: Iterable[AssetReference], project_directory: str | Path
) -> tuple[AssetCheck, ...]:
    checks = validate_assets(assets, project_directory)
    bad = [check for check in checks if check.status is not AssetStatus.OK]
    if bad:
        details = "; ".join(f"{c.asset.path}: {c.status.value}" for c in bad)
        raise AssetIntegrityError(f"Project asset validation failed: {details}.")
    return checks


def normalize_asset_path(path: object) -> str:
    if not isinstance(path, str):
        raise TypeError("Asset path must be a string.")
    raw = path.strip().replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (
        not raw
        or candidate.is_absolute()
        or candidate.anchor
        or any(part in ("", ".", "..") for part in candidate.parts)
        or (candidate.parts and ":" in candidate.parts[0])
    ):
        raise UnsafeAssetPathError(
            "Asset path must be a normalized relative path without '..'."
        )
    return candidate.as_posix()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _uuid(value: object) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Asset id must be a UUID.") from exc


__all__ = [
    "AssetCheck",
    "AssetReference",
    "AssetStatus",
    "normalize_asset_path",
    "require_valid_assets",
    "sha256_file",
    "validate_assets",
]
