"""Initialize headless plotting and safe stdio before dependency runtime hooks."""

import os
from pathlib import Path
import sys
import tempfile

os.environ["MPLBACKEND"] = "Agg"

if sys.stdout is None or sys.stderr is None:
    log_path = os.environ.get("OPTIPROP_PACKAGING_LOG")
    if log_path is None:
        log_path = str(Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
                       / "OptiProp" / "logs" / "launcher.log")
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_stream = open(log_path, "a", encoding="utf-8", buffering=1)
    except OSError:
        log_stream = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = log_stream
    if sys.stderr is None:
        sys.stderr = log_stream
if sys.stdin is None:
    sys.stdin = open(os.devnull, "r", encoding="utf-8")
