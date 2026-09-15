"""Actual Qt-free execution, file round trips and loopback security boundary."""

from copy import deepcopy
import http.client
import json
from pathlib import Path
import struct
from threading import Event, Thread
import time

import numpy as np
import pytest
from scipy.io import loadmat
import torch

from optiprop.io import load_field, read_zbf
from optiprop.workbench.app import WorkbenchServer
from optiprop.workbench.service import WorkbenchService, build_project


def example():
    from optiprop.workbench import app
    path = Path(app.__file__).parent / "examples/metalens_focus_binary2.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["source"]["grid"].update(nx=32, ny=32, dx=2e-6, dy=2e-6)
    return doc


def wait(service, identifier):
    deadline = time.monotonic() + 30
    while service.status(identifier)["status"] in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(.01)
    return service.status(identifier)


@pytest.fixture
def run(tmp_path):
    torch.set_num_threads(2)
    service = WorkbenchService(tmp_path / "output")
    doc = example()
    identifier = service.start({"project": doc})["id"]
    assert wait(service, identifier)["status"] == "completed"
    selection = {"job": identifier, "layer_id": doc["optical_system"]["layers"][0]["id"], "side": "after"}
    try:
        yield service, doc, selection
    finally:
        service.close()


def test_real_run_shared_ex_ey_and_selected_planes(run):
    service, doc, selection = run
    before = service.observed({**selection, "side": "before"})
    after = service.observed(selection)
    assert before.z_m == after.z_m == 0
    assert not torch.allclose(before.data, after.data)
    torch.testing.assert_close(after.data[1], .5 * after.data[0])
    propagated = service.observed({**selection, "layer_id": doc["optical_system"]["layers"][1]["id"]})
    assert propagated.z_m == pytest.approx(100e-6)
    assert propagated.dtype == torch.complex128
    original = deepcopy(service.status(selection["job"])["project"])
    doc["optical_system"]["layers"].clear()
    assert service.status(selection["job"])["project"] == original
    assert service.view(selection)["png"].startswith("iVBOR")


@pytest.mark.parametrize("kind", ["npz", "mat", "zbf", "png"])
def test_exports_have_requested_arrays_units_and_roundtrip(run, kind):
    service, _, selection = run
    field = service.observed(selection)
    path = Path(service.export({**selection, "format": kind})["path"])
    assert path.stat().st_size > 0
    assert path.with_suffix(path.suffix + ".json").is_file()
    if kind == "png":
        assert path.read_bytes().startswith(b"\x89PNG")
    else:
        loaded = load_field(path)
        torch.testing.assert_close(loaded.data, field.data)
    if kind == "mat":
        data = loadmat(path)
        for key in ("EX", "EY"):
            assert data[key].dtype == np.complex128
            assert data[key].shape == (32, 32)
        assert data["Nx"].item() == 32 and data["Ny"].item() == 32
        assert data["dx"].item() == pytest.approx(2e-6)
    if kind == "zbf":
        assert read_zbf(path).source_unit == 0
        raw = path.read_bytes()
        assert struct.unpack_from("<i", raw, 16)[0] == 0
        assert struct.unpack_from("<d", raw, 36)[0] == pytest.approx(2e-3)


def test_xz_uses_selected_plane_and_keeps_both_components(run):
    service, _, selection = run
    identifier = service.start({**selection, "count": 5, "distance_m": 20e-6}, "xz")["id"]
    status = wait(service, identifier)
    assert status["status"] == "completed", status
    arrays = np.load(Path(status["output_dir"]) / "xz.npz")
    assert arrays["intensity"].shape == (2, 5, 32)
    np.testing.assert_allclose(arrays["intensity"][1], .25 * arrays["intensity"][0])
    assert arrays["z_m"][-1] == pytest.approx(20e-6)


def test_reimport_exported_source_preserves_data_and_asset_hash(run):
    service, _, selection = run
    path = service.export({**selection, "format": "mat"})["path"]
    imported = service.import_source(path)
    _, _, field = build_project(imported["project"], imported["asset_root"])
    torch.testing.assert_close(field.data, service.observed(selection).data)
    with (Path(imported["asset_root"]) / "source.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(Exception, match="changed"):
        build_project(imported["project"], imported["asset_root"])


@pytest.mark.parametrize("change", [
    lambda d: d["source"]["grid"].update(nx=100000),
    lambda d: d["source"]["grid"].update(ny=1),
    lambda d: d["optical_system"]["layers"][1]["spec"]["padding"].update(factor=100),
    lambda d: d["optical_system"]["layers"][0].update(coefficients=[0]*33),
])
def test_web_resource_limits_fail_before_large_allocation(change, tmp_path):
    doc = example(); change(doc)
    with pytest.raises(ValueError):
        build_project(doc, tmp_path)


def test_failed_run_is_reported_not_marked_complete(tmp_path):
    service = WorkbenchService(tmp_path)
    try:
        doc = example(); doc["optical_system"]["layers"][0]["coefficients"] = []
        job = service.start({"project": doc})["id"]
        assert wait(service, job)["status"] == "failed"
    finally:
        service.close()


def test_zbf_refuses_non_power_of_two(run):
    service, doc, _ = run
    doc["source"]["grid"]["nx"] = 31
    job = service.start({"project": doc})["id"]
    assert wait(service, job)["status"] == "completed"
    with pytest.raises(ValueError, match="powers of two"):
        service.export({"job": job, "layer_id": doc["source"]["id"], "format": "zbf"})


@pytest.mark.parametrize("cancel", [False, True])
def test_last_good_run_survives_xz_then_failed_or_cancelled_replacement(run, monkeypatch, cancel):
    service, doc, selection = run
    for _ in range(2):
        scan = service.start({**selection, "count": 3, "distance_m": 10e-6}, "xz")["id"]
        assert wait(service, scan)["status"] == "completed"
    entered, release = Event(), Event()
    import optiprop.workbench.service as module
    real_builder = module.build_project
    def delayed(*args):
        entered.set()
        assert release.wait(10)
        return real_builder(*args)
    monkeypatch.setattr(module, "build_project", delayed)
    replacement = deepcopy(doc)
    if not cancel:
        replacement["optical_system"]["layers"][0]["coefficients"] = []
    identifier = service.start({"project": replacement})["id"]
    try:
        assert entered.wait(10)
        with pytest.raises(ValueError, match="task is running"):
            service.start({"project": doc})
        if cancel:
            service.cancel(identifier)
    finally:
        release.set()
    assert wait(service, identifier)["status"] == ("cancelled" if cancel else "failed")
    assert service.last_good_run == selection["job"]
    assert Path(service.export({**selection, "format": "npz"})["path"]).is_file()


def test_loopback_auth_host_origin_and_no_arbitrary_file_serving(tmp_path):
    with WorkbenchServer(tmp_path) as server:
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        def request(path, headers=None, method="GET", body=None):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse(); data = response.read(); conn.close()
            return response.status, data
        try:
            assert server.server_address[0] == "127.0.0.1"
            assert request("/")[0] == 200
            assert request("/api/bootstrap")[0] == 403
            auth = {"X-OptiProp-Token": server.token}
            code, data = request("/api/bootstrap", auth)
            assert code == 200 and len(json.loads(data)["examples"]) == 4
            assert request("/api/bootstrap", {**auth, "Origin": "https://attacker.invalid"})[0] == 403
            assert request("/api/bootstrap", {**auth, "Host": "attacker.invalid"})[0] == 403
            assert request("/../../pyproject.toml", auth)[0] == 404
            assert request("/api/run", {**auth, "Content-Type": "application/json"}, "POST", "[]")[0] == 400
            assert request("/api/run", {**auth, "Content-Type": "application/json", "Content-Length": "99999999"}, "POST", "{}")[0] == 400
        finally:
            server.shutdown(); thread.join(timeout=10)
