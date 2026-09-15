from __future__ import annotations

from optiprop.gui.models import DEFAULT_SOURCE, new_layer
from optiprop.gui.simulation import SimulationCache, SimulationOutput, build_source, build_system, simulation_key


def _source():
    return {
        **DEFAULT_SOURCE,
        "nx": 16, "ny": 12, "dx_m": 8e-6, "dy_m": 9e-6,
        "waist_x_m": 4e-5, "waist_y_m": 4e-5,
    }


def test_dict_builders_execute_real_optical_system():
    propagation = new_layer("propagation")
    propagation.update({"method": "asm", "distance_m": 1e-3, "padding": "none"})
    field = build_source(_source()).create()
    result = build_system([propagation]).execute(field)
    assert result.final_field.grid.shape == (12, 16)
    assert result.final_field.z_m == 1e-3


def test_cache_is_lru_and_key_tracks_asset_state(tmp_path):
    asset = tmp_path / "field.npz"
    asset.write_bytes(b"first")
    config = {"source": {"path": str(asset)}, "layers": []}
    first = simulation_key(config)
    asset.write_bytes(b"changed-size")
    assert simulation_key(config) != first
    cache = SimulationCache(1)
    one = SimulationOutput(object())
    two = SimulationOutput(object())
    cache.put("one", one); cache.put("two", two)
    assert cache.get("one") is None
    assert cache.get("two") is two
