"""Compatibility wrapper for `evaluation.evaluators.evaluate_filtered_reference`."""
from __future__ import annotations

import importlib as _importlib
import runpy as _runpy
import sys as _sys
from pathlib import Path as _Path

_SRC = _Path(__file__).resolve().parents[1]
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

_TARGET = "evaluation.evaluators.evaluate_filtered_reference"

if __name__ == "__main__":
    _runpy.run_module(_TARGET, run_name="__main__")
else:
    _module = _importlib.import_module(_TARGET)
    for _name, _value in vars(_module).items():
        if _name not in {
            "__builtins__",
            "__cached__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__spec__",
        }:
            globals()[_name] = _value
