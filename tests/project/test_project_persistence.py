import json
from uuid import uuid4

import pytest

from optiprop.core import Grid2D
from optiprop.project import (
    ComputeSettings,
    OptiPropProject,
    ProjectFormatError,
    UnsupportedProjectVersionError,
    load_project,
    migrate_project_document,
    project_to_domain,
    save_project,
)
from optiprop.propagation import PaddingSpec, PropagationMethod, PropagationSpec
from optiprop.system import (
    ApertureShape,
    ApertureSpec,
    IdealLensLayer,
    IncidentSource,
    InterfaceLayer,
    OpticalSystem,
    PropagationLayer,
)


def project_fixture():
    grid = Grid2D(24, 16, 2e-6, 3e-6, 1e-6, -2e-6)
    source = IncidentSource.gaussian(
        grid,
        532e-9,
        18e-6,
        aperture=ApertureSpec(ApertureShape.CIRCLE, 60e-6),
        name="Input beam",
    )
    system = OpticalSystem(
        (
            IdealLensLayer(0.02, design_wavelength_m=532e-9, name="L1"),
            InterfaceLayer(1.0, 1.5, name="Glass entrance"),
            PropagationLayer(
                PropagationSpec(
                    PropagationMethod.BLAS,
                    0.001,
                    medium_index=1.5,
                    padding=PaddingSpec.by_factor(2),
                    options={"bandlimit": True},
                ),
                name="Glass",
            ),
        ),
        name="Bench",
    )
    return OptiPropProject.from_domain(
        source,
        system,
        name="Round trip",
        compute_settings=ComputeSettings(device="cpu", max_cache_bytes=12345),
        ui_state={"selected_layer": str(system.layers[1].id), "splitter": [200, 600]},
    )


def test_json_roundtrip_is_stable_and_reconstructs_domain(tmp_path):
    project = project_fixture()
    path = tmp_path / "design.optiprop.json"
    save_project(project, path)
    loaded = load_project(path)

    assert loaded.to_dict() == project.to_dict()
    assert loaded.id == project.id
    source, system = project_to_domain(loaded)
    assert str(source.id) == loaded.source["id"]
    assert [str(layer.id) for layer in system.layers] == [
        item["id"] for item in loaded.optical_system["layers"]
    ]
    assert system.layers[2].spec.method is PropagationMethod.BLAS
    assert system.layers[2].spec.padding.factor == 2


def test_save_replaces_existing_file_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "project.json"
    path.write_text("old", encoding="utf-8")
    save_project(project_fixture(), path)
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "optiprop-project"
    assert not list(tmp_path.glob(".project.json.*.tmp"))


def test_migrates_legacy_v1_property_names(tmp_path):
    current = project_fixture().to_dict()
    current["version"] = current.pop("schema_version")
    current["id"] = current.pop("project_id")
    current["system"] = current.pop("optical_system")
    current.pop("format")
    migrated = migrate_project_document(current)
    assert migrated["schema_version"] == 1
    assert migrated["project_id"]
    assert migrated["optical_system"]["type"] == "OpticalSystem"


@pytest.mark.parametrize("version", [0, 2, 999])
def test_unknown_project_version_is_rejected(version):
    document = project_fixture().to_dict()
    document["schema_version"] = version
    with pytest.raises(UnsupportedProjectVersionError):
        migrate_project_document(document)


def test_arbitrary_json_is_not_guessed_as_legacy_project():
    with pytest.raises(ProjectFormatError, match="schema_version"):
        migrate_project_document({"id": str(uuid4()), "name": "not enough"})
