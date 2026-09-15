"""Project persistence errors with user-facing failure categories."""


class ProjectError(Exception):
    """Base class for project model and persistence failures."""


class ProjectFormatError(ProjectError):
    """Raised when a project document is malformed."""


class UnsupportedProjectVersionError(ProjectFormatError):
    """Raised when a project uses a newer or unknown schema version."""


class UnsafeAssetPathError(ProjectFormatError):
    """Raised when an asset path could escape the project directory."""


class AssetIntegrityError(ProjectError):
    """Raised when a required project asset is missing or has changed."""


__all__ = [
    "AssetIntegrityError",
    "ProjectError",
    "ProjectFormatError",
    "UnsafeAssetPathError",
    "UnsupportedProjectVersionError",
]
