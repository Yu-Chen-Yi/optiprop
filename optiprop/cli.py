"""Small command-line companions for the OptiProp desktop application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import inspect_field, load_field, save_field


def inspect_main(argv: list[str] | None = None) -> int:
    """Inspect an NPZ, MAT, or ZBF field without loading it into the GUI."""

    parser = argparse.ArgumentParser(prog="optiprop-inspect")
    parser.add_argument("field", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        report = inspect_field(args.field)
    except Exception as exc:
        parser.exit(2, f"optiprop-inspect: {exc}\n")

    payload = {
        "path": str(report.path),
        "format": report.format.value,
        "canonical": report.is_canonical,
        "schema": report.schema_name,
        "schema_version": report.schema_version,
        "keys": list(report.keys),
        "arrays": [
            {
                "key": item.key,
                "shape": list(item.shape),
                "dtype": item.dtype,
                "complex": item.is_complex,
                "finite": item.is_finite,
            }
            for item in report.arrays
        ],
        "metadata": dict(report.scalar_metadata),
        "warnings": [issue.message for issue in report.warnings],
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"{payload['format'].upper()}  {payload['path']}")
        print(f"canonical: {payload['canonical']}")
        if payload["schema"]:
            print(f"schema: {payload['schema']} v{payload['schema_version']}")
        for item in payload["arrays"]:
            print(
                f"{item['key']}: shape={tuple(item['shape'])} "
                f"dtype={item['dtype']} complex={item['complex']}"
            )
        for key, value in payload["metadata"].items():
            print(f"{key}: {value}")
    return 0


def convert_main(argv: list[str] | None = None) -> int:
    """Convert a canonical field between NPZ, MAT, and ZBF containers."""

    parser = argparse.ArgumentParser(prog="optiprop-convert")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="permit harmless container additions while reading",
    )
    args = parser.parse_args(argv)
    try:
        field = load_field(args.source, strict=not args.non_strict)
        report = save_field(field, args.destination)
    except Exception as exc:
        parser.exit(2, f"optiprop-convert: {exc}\n")
    print(
        f"wrote {report.path} ({report.bytes_written} bytes, "
        f"sha256={report.sha256})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(inspect_main())
