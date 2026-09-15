"""Factories between project configs and the numerical domain model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID

import torch

from ..core import Field2D, Grid2D
from ..propagation import PaddingSpec, PropagationSpec
from ..system import (
    ApertureLayer,
    ApertureSpec,
    Binary2LensLayer,
    ComplexMaskLayer,
    IdealLensLayer,
    IncidentSource,
    InterfaceLayer,
    OpticalSystem,
    PropagationLayer,
    SourceKind,
)
from .assets import AssetReference
from .models import OptiPropProject

AssetLoader = Callable[[AssetReference], Field2D | torch.Tensor]


def project_to_domain(
    project: OptiPropProject,
    *,
    asset_loader: AssetLoader | None = None,
) -> tuple[IncidentSource, OpticalSystem]:
    """Rebuild executable source/system objects, preserving every UUID."""

    if not isinstance(project, OptiPropProject):
        raise TypeError("project must be an OptiPropProject.")
    source = source_from_config(project.source, project, asset_loader)
    system = optical_system_from_config(
        project.optical_system, project, asset_loader
    )
    return source, system


def source_from_config(
    config: Mapping[str, Any],
    project: OptiPropProject | None = None,
    asset_loader: AssetLoader | None = None,
) -> IncidentSource:
    data = dict(config)
    if data.pop("type", "IncidentSource") != "IncidentSource":
        raise ValueError("Unsupported source type.")
    data.pop("version", None)
    imported_summary = data.pop("imported_field", None)
    asset_id = data.pop("asset_id", None)
    data["id"] = UUID(str(data["id"]))
    data["kind"] = SourceKind(data["kind"])
    data["medium_index"] = _complex(data.get("medium_index", 1.0))
    data["components"] = tuple(data.get("components", ("scalar",)))
    data["amplitudes"] = tuple(_complex(item) for item in data.get("amplitudes", [1]))
    aperture = data.get("aperture")
    data["aperture"] = None if aperture is None else ApertureSpec(**dict(aperture))
    grid = data.get("grid")
    data["grid"] = None if grid is None else Grid2D(**dict(grid))
    if data["kind"] is SourceKind.IMPORTED_FIELD:
        if asset_id is None:
            raise ValueError("Imported source config requires asset_id.")
        loaded = _load_asset(project, asset_loader, asset_id)
        if not isinstance(loaded, Field2D):
            raise TypeError("An imported source asset loader must return Field2D.")
        data["source_field"] = loaded
    elif imported_summary is not None:
        raise ValueError("Only imported_field sources may contain imported_field data.")
    return IncidentSource(**data)


def optical_system_from_config(
    config: Mapping[str, Any],
    project: OptiPropProject | None = None,
    asset_loader: AssetLoader | None = None,
) -> OpticalSystem:
    data = dict(config)
    if data.get("type") != "OpticalSystem":
        raise ValueError("Unsupported optical system type.")
    layers = tuple(
        layer_from_config(item, project, asset_loader)
        for item in data.get("layers", ())
    )
    return OpticalSystem(
        layers=layers,
        id=UUID(str(data["id"])),
        name=data.get("name", "Optical system"),
    )


def layer_from_config(
    config: Mapping[str, Any],
    project: OptiPropProject | None = None,
    asset_loader: AssetLoader | None = None,
):
    data = dict(config)
    layer_type = data.pop("type", None)
    data.pop("version", None)
    common = {
        "id": UUID(str(data.pop("id"))),
        "name": data.pop("name"),
        "enabled": data.pop("enabled", True),
    }
    if layer_type == "PropagationLayer":
        spec = _propagation_spec(dict(data.pop("spec")))
        return PropagationLayer(
            spec=spec,
            keep_intermediate=data.pop("keep_intermediate", True),
            **common,
        )
    if layer_type == "ApertureLayer":
        return ApertureLayer(aperture=ApertureSpec(**dict(data.pop("aperture"))), **common)
    if layer_type == "IdealLensLayer":
        aperture = data.pop("aperture", None)
        data["aperture"] = None if aperture is None else ApertureSpec(**dict(aperture))
        data["medium_index"] = _optional_complex(data.get("medium_index"))
        return IdealLensLayer(**data, **common)
    if layer_type == "Binary2LensLayer":
        aperture = data.pop("aperture", None)
        data["aperture"] = None if aperture is None else ApertureSpec(**dict(aperture))
        return Binary2LensLayer(**data, **common)
    if layer_type == "InterfaceLayer":
        data["n1"] = _complex(data["n1"])
        data["n2"] = _complex(data["n2"])
        return InterfaceLayer(**data, **common)
    if layer_type == "ComplexMaskLayer":
        asset_id = data.pop("asset_id", None)
        data.pop("transmission", None)
        if asset_id is None:
            raise ValueError("ComplexMaskLayer config requires asset_id.")
        loaded = _load_asset(project, asset_loader, asset_id)
        transmission = loaded.data if isinstance(loaded, Field2D) else loaded
        if not isinstance(transmission, torch.Tensor):
            raise TypeError("A mask asset loader must return Field2D or torch.Tensor.")
        data["transmission"] = transmission
        data["grid"] = Grid2D(**dict(data["grid"]))
        return ComplexMaskLayer(**data, **common)
    raise ValueError(f"Unsupported layer type {layer_type!r}.")


def _propagation_spec(data: dict[str, Any]) -> PropagationSpec:
    padding_data = dict(data.pop("padding", {}))
    padding = PaddingSpec(
        mode=padding_data.get("mode", "auto"),
        factor=padding_data.get("factor"),
        shape=padding_data.get("shape"),
    )
    output_grid = data.pop("output_grid", None)
    data["output_grid"] = None if output_grid is None else Grid2D(**dict(output_grid))
    data["medium_index"] = _optional_complex(data.get("medium_index"))
    return PropagationSpec(padding=padding, **data)


def _load_asset(
    project: OptiPropProject | None,
    loader: AssetLoader | None,
    asset_id: object,
) -> Field2D | torch.Tensor:
    if project is None or loader is None:
        raise ValueError("Asset-backed objects require project and asset_loader.")
    return loader(project.asset(str(asset_id)))


def _complex(value: object) -> complex:
    if isinstance(value, Mapping):
        return complex(value.get("real", 0.0), value.get("imag", 0.0))
    return complex(value)


def _optional_complex(value: object) -> complex | None:
    return None if value is None else _complex(value)


__all__ = [
    "AssetLoader",
    "layer_from_config",
    "optical_system_from_config",
    "project_to_domain",
    "source_from_config",
]
