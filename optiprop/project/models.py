"""Immutable versioned OptiProp project model."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4

from ..system import IncidentSource, OpticalSystem
from .assets import AssetReference

PROJECT_FORMAT = "optiprop-project"
CURRENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ComputeSettings:
    """Serializable execution preferences; no runtime backend objects."""

    device: str = "auto"
    precision: str = "inherit"
    cache_enabled: bool = True
    max_cache_bytes: int = 512 * 1024 * 1024
    continue_on_error: bool = False

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.precision not in ("inherit", "complex64", "complex128"):
            raise ValueError(
                "precision must be 'inherit', 'complex64', or 'complex128'."
            )
        if not isinstance(self.cache_enabled, bool):
            raise TypeError("cache_enabled must be a bool.")
        if isinstance(self.max_cache_bytes, bool) or int(self.max_cache_bytes) < 0:
            raise ValueError("max_cache_bytes must be non-negative.")
        object.__setattr__(self, "max_cache_bytes", int(self.max_cache_bytes))
        if not isinstance(self.continue_on_error, bool):
            raise TypeError("continue_on_error must be a bool.")

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "precision": self.precision,
            "cache_enabled": self.cache_enabled,
            "max_cache_bytes": self.max_cache_bytes,
            "continue_on_error": self.continue_on_error,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ComputeSettings":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("compute_settings must be an object.")
        return cls(**value)


@dataclass(frozen=True)
class OptiPropProject:
    """Complete, portable GUI project independent of PySide."""

    source: Mapping[str, Any]
    optical_system: Mapping[str, Any]
    name: str = "Untitled project"
    id: UUID = field(default_factory=uuid4)
    compute_settings: ComputeSettings = field(default_factory=ComputeSettings)
    ui_state: Mapping[str, Any] = field(default_factory=dict)
    assets: tuple[AssetReference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id))
        name = str(self.name).strip()
        if not name:
            raise ValueError("Project name must not be empty.")
        object.__setattr__(self, "name", name)
        if not isinstance(self.compute_settings, ComputeSettings):
            raise TypeError("compute_settings must be ComputeSettings.")
        object.__setattr__(self, "source", _freeze_json_object(self.source, "source"))
        object.__setattr__(
            self,
            "optical_system",
            _freeze_json_object(self.optical_system, "optical_system"),
        )
        object.__setattr__(self, "ui_state", _freeze_json_object(self.ui_state, "ui_state"))
        object.__setattr__(self, "metadata", _freeze_json_object(self.metadata, "metadata"))
        assets = tuple(self.assets)
        if not all(isinstance(item, AssetReference) for item in assets):
            raise TypeError("assets must contain AssetReference objects.")
        ids = [item.id for item in assets]
        if len(ids) != len(set(ids)):
            raise ValueError("Asset IDs must be unique within a project.")
        object.__setattr__(self, "assets", assets)

    @classmethod
    def from_domain(
        cls,
        source: IncidentSource,
        optical_system: OpticalSystem,
        *,
        source_asset_id: UUID | str | None = None,
        mask_asset_ids: Mapping[UUID | str, UUID | str] | None = None,
        **kwargs: Any,
    ) -> "OptiPropProject":
        if not isinstance(source, IncidentSource):
            raise TypeError("source must be an IncidentSource.")
        if not isinstance(optical_system, OpticalSystem):
            raise TypeError("optical_system must be an OpticalSystem.")
        source_config = source.to_config()
        if source.source_field is not None:
            if source_asset_id is None:
                raise ValueError("Imported sources require source_asset_id.")
            source_config["asset_id"] = str(_uuid(source_asset_id))
        system_config = optical_system.to_config()
        by_layer = {
            str(_uuid(layer_id)): str(_uuid(asset_id))
            for layer_id, asset_id in (mask_asset_ids or {}).items()
        }
        for layer in system_config["layers"]:
            if layer["type"] == "ComplexMaskLayer":
                layer_id = layer["id"]
                if layer_id not in by_layer:
                    raise ValueError(
                        f"Complex mask layer {layer_id} requires an asset ID."
                    )
                layer["asset_id"] = by_layer[layer_id]
        return cls(
            source=source_config,
            optical_system=system_config,
            **kwargs,
        )

    def asset(self, asset_id: UUID | str) -> AssetReference:
        wanted = _uuid(asset_id)
        for item in self.assets:
            if item.id == wanted:
                return item
        raise KeyError(str(wanted))

    def to_dict(self) -> dict[str, object]:
        return {
            "format": PROJECT_FORMAT,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "project_id": str(self.id),
            "name": self.name,
            "source": _thaw(self.source),
            "optical_system": _thaw(self.optical_system),
            "compute_settings": self.compute_settings.to_dict(),
            "ui_state": _thaw(self.ui_state),
            "assets": [asset.to_dict() for asset in self.assets],
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> "OptiPropProject":
        if not isinstance(value, dict):
            raise TypeError("Project document must be an object.")
        return cls(
            id=value["project_id"],
            name=value["name"],
            source=value["source"],
            optical_system=value["optical_system"],
            compute_settings=ComputeSettings.from_dict(value.get("compute_settings")),
            ui_state=value.get("ui_state", {}),
            assets=tuple(AssetReference.from_dict(item) for item in value.get("assets", [])),
            metadata=value.get("metadata", {}),
        )


def _uuid(value: object) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Project id must be a UUID.") from exc


def _freeze_json_object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    frozen = _freeze(dict(value))
    json.dumps(_thaw(frozen), allow_nan=False, sort_keys=True)
    return frozen


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Project object keys must be strings.")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Project numeric values must be finite.")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported project value {type(value).__name__}.")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "PROJECT_FORMAT",
    "ComputeSettings",
    "OptiPropProject",
]
