"""Offline environment diagnostics. Repair is an explicit command, never an import side effect."""

from __future__ import annotations

import importlib
import json
import sys

REPAIR = 'python -m pip install "numpy==1.26.4" "pyarrow==20.0.0" "datasets==3.6.0"'


def check() -> dict:
    versions, problems = {}, []
    for name in ("numpy", "pyarrow", "datasets"):
        try:
            module = importlib.import_module(name)
            versions[name] = module.__version__
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
    if not problems:
        try:
            import numpy as np
            import pyarrow as pa
            from datasets import Dataset

            array = pa.array(np.array([1, 2], dtype=np.int64))
            assert Dataset.from_dict({"value": array.to_pylist()})[0]["value"] == 1
        except Exception as exc:
            problems.append(f"arrow_numpy_incompatible: {type(exc).__name__}: {exc}")
    return {
        "status": "ok" if not problems else "dependency_incompatible",
        "python": sys.version.split()[0],
        "versions": versions,
        "errors": problems,
        "tested_python": "3.11",
        "repair_command": REPAIR,
    }


def require() -> None:
    report = check()
    if report["errors"]:
        raise RuntimeError(json.dumps(report))


if __name__ == "__main__":
    report = check()
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))
