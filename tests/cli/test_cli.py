from __future__ import annotations

import json

import torch

from optiprop.cli import convert_main, inspect_main
from optiprop.core import Field2D, Grid2D
from optiprop.io import load_field, save_field


def _field() -> Field2D:
    data = torch.ones((1, 5, 7), dtype=torch.complex64)
    return Field2D(data, Grid2D(7, 5, 2e-6, 3e-6), 532e-9)


def test_inspect_json(tmp_path, capsys):
    path = tmp_path / "field.npz"
    save_field(_field(), path)

    assert inspect_main([str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "npz"
    assert payload["canonical"] is True
    data_item = next(item for item in payload["arrays"] if item["key"] == "field")
    assert data_item["shape"] == [1, 5, 7]


def test_convert_between_canonical_formats(tmp_path):
    source = tmp_path / "field.npz"
    destination = tmp_path / "field.mat"
    save_field(_field(), source)

    assert convert_main([str(source), str(destination)]) == 0
    loaded = load_field(destination)
    assert loaded.grid == _field().grid
    assert torch.equal(loaded.data, _field().data)
