"""Qt-free desktop launcher; authenticated loopback-only HTTP, not a web host."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import sys
from threading import Thread
from urllib.parse import urlsplit
import webbrowser

from .service import WorkbenchService

_ROOT = Path(__file__).resolve().parent


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, output_root, *, port=0, initial_project=None, asset_root=None):
        super().__init__(("127.0.0.1", port), WorkbenchHandler)
        self.token = secrets.token_urlsafe(32)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.service = WorkbenchService(output_root)
        self.initial_project = initial_project
        self.asset_root = str(Path(asset_root or Path.cwd()).resolve())

    def server_close(self):
        super().server_close()
        if hasattr(self, "service"):
            self.service.close()


class WorkbenchHandler(BaseHTTPRequestHandler):
    # No SimpleHTTPRequestHandler: never expose arbitrary files or directories.
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        pass  # URLs may include session secrets; no request logging.

    def _allowed(self, *, authenticated=False):
        if self.headers.get("Host") != urlsplit(self.server.origin).netloc:
            self._json({"error": "Invalid Host"}, 403); return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != self.server.origin:
            self._json({"error": "Cross-origin access is not permitted"}, 403); return False
        if authenticated and not secrets.compare_digest(self.headers.get("X-OptiProp-Token", ""), self.server.token):
            self._json({"error": "Session expired or unauthorized"}, 403); return False
        return True

    def _send(self, payload, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, payload, code=200):
        self._send(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", code)

    def do_GET(self):
        path = urlsplit(self.path).path
        if not self._allowed(authenticated=path.startswith("/api/")):
            return
        try:
            if path in ("/", "/app.js", "/app.css"):
                name = {"/": "index.html", "/app.js": "app.js", "/app.css": "app.css"}[path]
                mime = {"index.html": "text/html", "app.js": "text/javascript", "app.css": "text/css"}[name]
                self._send((_ROOT / "static" / name).read_bytes(), mime + "; charset=utf-8")
            elif path == "/api/bootstrap":
                from .. import __version__
                self._json({"version": __version__, "examples": {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((_ROOT / "examples").glob("*.json"))},
                            "initial_project": self.server.initial_project, "asset_root": self.server.asset_root,
                            "output_root": str(self.server.service.output_root)})
            elif path.startswith("/api/jobs/"):
                self._json(self.server.service.status(path.rsplit("/", 1)[-1]))
            else:
                self._json({"error": "Not found"}, 404)
        except KeyError:
            self._json({"error": "Run not found or no longer held in memory"}, 404)
        except Exception as exc:
            logging.exception("Workbench GET failed")
            self._json({"error": str(exc)}, 400)

    def do_POST(self):
        if not self._allowed(authenticated=True):
            return
        try:
            if "Transfer-Encoding" in self.headers:
                raise ValueError("Chunked requests are not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2 * 1024**2:
                raise ValueError("Request must contain at most 2 MiB of JSON")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json")
            payload = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            path = urlsplit(self.path).path
            service = self.server.service
            if path == "/api/validate":
                result = service.validate(payload["project"], payload.get("asset_root", self.server.asset_root))
            elif path == "/api/run":
                result = service.start(payload)
            elif path == "/api/xz":
                result = service.start(payload, "xz")
            elif path == "/api/cancel":
                result = service.cancel(payload["job"])
            elif path == "/api/view":
                result = service.view(payload)
            elif path == "/api/export":
                result = service.export(payload)
            elif path == "/api/xz-image":
                job = service.jobs[payload["job"]]
                if job["kind"] != "xz" or job["status"] != "completed":
                    raise ValueError("XZ scan is not complete")
                result = {"png": base64.b64encode(job["png"]).decode("ascii")}
            elif path == "/api/import-source":
                result = service.import_source(payload["path"])
            elif path == "/api/shutdown":
                result = {"stopping": True}
                Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._json({"error": "Not found"}, 404); return
            self._json(result)
        except KeyError as exc:
            self._json({"error": f"Missing field or expired run: {exc}"}, 400)
        except Exception as exc:
            logging.exception("Workbench action failed")
            self._json({"error": str(exc)}, 400)


def self_test(output):
    """Frozen executable smoke test: actual Ex/Ey propagation, plots and formats."""
    import time
    import torch
    from ..io import load_field, read_zbf
    from scipy.io import loadmat
    example = json.loads((_ROOT / "examples/metalens_focus_binary2.json").read_text(encoding="utf-8"))
    example["source"]["grid"].update(nx=64, ny=64, dx=1e-6, dy=1e-6)
    service = WorkbenchService(output)
    try:
        job = service.start({"project": example})["id"]
        deadline = time.monotonic() + 120
        while service.status(job)["status"] in ("queued", "running"):
            if time.monotonic() > deadline:
                raise TimeoutError("Self-test run exceeded 120 seconds")
            time.sleep(.05)
        if service.status(job)["status"] != "completed":
            raise RuntimeError(service.status(job)["message"])
        selection = {"job": job, "layer_id": example["optical_system"]["layers"][-1]["id"], "side": "after"}
        assert service.view(selection)["png"]
        for kind in ("npz", "mat", "zbf", "png"):
            path = Path(service.export({**selection, "format": kind})["path"])
            assert path.stat().st_size > 0
            if kind == "npz":
                assert torch.isfinite(load_field(path).data).all()
            if kind == "mat":
                assert loadmat(path)["EX"].dtype.name == "complex128"
            if kind == "zbf":
                beam = read_zbf(path)
                assert beam.source_unit == 0 and beam.nx == 64
        if any(name.startswith(("PySide", "PyQt", "shiboken", "tkinter")) for name in sys.modules):
            raise RuntimeError("A removed GUI toolkit was imported")
        result = {"status": "passed", "torch": torch.__version__, "cuda_build": torch.version.cuda, "run": service.status(job)}
        (Path(output) / "self-test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    finally:
        service.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="OptiProp local browser workbench (no Qt)")
    parser.add_argument("project", nargs="?", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Documents/OptiProp/output")
    parser.add_argument("--session-file", type=Path, help="Write local session connection details for automated testing")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output_dir / "workbench.log", level=logging.INFO, encoding="utf-8")
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.self_test:
        return self_test(args.output_dir)
    initial = None
    if args.project:
        initial = json.loads(args.project.read_text(encoding="utf-8-sig"))
        if initial.get("format") != "optiprop-project":
            parser.error("This is a legacy Qt project; use a canonical project from Example/projects instead.")
    with WorkbenchServer(args.output_dir, port=args.port, initial_project=initial,
                          asset_root=args.project.parent if args.project else None) as server:
        url = server.origin + "/#" + server.token
        if args.session_file:
            args.session_file.write_text(json.dumps({"url": url, "origin": server.origin, "token": server.token}), encoding="utf-8")
        if sys.stdout is not None:
            print("OptiProp listening at " + server.origin + " (loopback only)", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
