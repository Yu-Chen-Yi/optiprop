from __future__ import annotations

from PySide6.QtCore import Qt

from optiprop.gui.models import LayerListModel, SourceModel, new_layer


def test_layer_model_add_toggle_move_remove(qapp):
    model = LayerListModel()
    first = model.append_kind("propagation")
    second = model.append_kind("lens")
    assert (first, second) == (0, 1)
    assert model.data(model.index(0), model.TypeRole) == "propagation"
    assert model.setData(model.index(0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert model.layer(0)["enabled"] is False
    assert model.move(1, -1) == 0
    assert model.layer(0)["type"] == "lens"
    assert model.remove_row(1)
    assert model.rowCount() == 1


def test_source_model_returns_defensive_copy(qapp):
    model = SourceModel()
    copied = model.config
    copied["nx"] = 2
    assert model.config["nx"] == 256


def test_new_layer_has_stable_unique_id():
    assert new_layer("lens")["id"] != new_layer("lens")["id"]
