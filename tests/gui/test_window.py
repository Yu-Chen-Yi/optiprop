from __future__ import annotations

from optiprop.gui.main_window import MainWindow
from optiprop.project import OptiPropProject


def test_main_window_offscreen_smoke(qapp):
    window = MainWindow()
    window.load_dict({
        "source": {**window.source_model.config, "nx": 16, "ny": 16},
        "layers": [],
        "xz": {"enabled": True, "method": "asm", "z_start_m": 0.0, "z_stop_m": 0.001, "count": 3},
    })
    window.show()
    qapp.processEvents()
    assert window.isVisible()
    assert window.project_dict()["xz"]["count"] == 3
    assert window.validate_project()
    window.close()


def test_layer_controls_update_serialized_project(qapp):
    window = MainWindow()
    window.add_kind.setCurrentIndex(window.add_kind.findData("lens"))
    window._add_layer()
    window.layer_editor.set_values(window.layer_model.layer(0))
    values = window.layer_model.layer(0)
    values["focal_length_m"] = 0.123
    window._update_layer(values)
    assert window.project_dict()["layers"][0]["focal_length_m"] == 0.123
    window.close()


def test_window_emits_and_loads_versioned_project(qapp):
    window = MainWindow()
    window.load_dict({
        "source": {**window.source_model.config, "nx": 12, "ny": 10},
        "layers": [],
        "xz": {"enabled": False},
    })
    document = window.persisted_project().to_dict()
    assert document["format"] == "optiprop-project"
    restored = OptiPropProject.from_dict(document)

    second = MainWindow()
    second.load_dict(restored.to_dict())
    assert second.source_model.config["nx"] == 12
    assert second.persisted_project().id == restored.id
    second.close()
    window.close()


def test_project_asset_paths_resolve_from_and_save_relative_to_project(qapp, tmp_path):
    asset = tmp_path / "field.mat"
    window = MainWindow()
    document = window.persisted_project().to_dict()
    document["source"] = {
        "kind": "imported_field",
        "path": asset.name,
        "mapping": {},
    }
    window.load_dict(document, base_dir=tmp_path)
    assert window.source_model.config["path"] == str(asset.resolve())

    window.project_path = tmp_path / "example.oprop"
    assert window.persisted_project().to_dict()["source"]["path"] == asset.name
    window.close()
