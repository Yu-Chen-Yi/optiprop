"""Versioned OptiProp project persistence and domain reconstruction."""

from .assets import (
    AssetCheck,
    AssetReference,
    AssetStatus,
    normalize_asset_path,
    require_valid_assets,
    sha256_file,
    validate_assets,
)
from .domain import (
    AssetLoader,
    layer_from_config,
    optical_system_from_config,
    project_to_domain,
    source_from_config,
)
from .errors import (
    AssetIntegrityError,
    ProjectError,
    ProjectFormatError,
    UnsafeAssetPathError,
    UnsupportedProjectVersionError,
)
from .models import (
    CURRENT_SCHEMA_VERSION,
    PROJECT_FORMAT,
    ComputeSettings,
    OptiPropProject,
)
from .persistence import load_project, migrate_project_document, save_project

__all__ = [
    "AssetCheck",
    "AssetIntegrityError",
    "AssetLoader",
    "AssetReference",
    "AssetStatus",
    "CURRENT_SCHEMA_VERSION",
    "ComputeSettings",
    "OptiPropProject",
    "PROJECT_FORMAT",
    "ProjectError",
    "ProjectFormatError",
    "UnsafeAssetPathError",
    "UnsupportedProjectVersionError",
    "layer_from_config",
    "load_project",
    "migrate_project_document",
    "normalize_asset_path",
    "optical_system_from_config",
    "project_to_domain",
    "require_valid_assets",
    "save_project",
    "sha256_file",
    "source_from_config",
    "validate_assets",
]
