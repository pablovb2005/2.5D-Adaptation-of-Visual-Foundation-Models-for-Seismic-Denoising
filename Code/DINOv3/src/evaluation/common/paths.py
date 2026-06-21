"""Path helpers for evaluation scripts.

The evaluation package is invoked both as modules and through compatibility
wrapper files such as ``python evaluation/evaluate.py``. These helpers avoid
hard-coded ``Path(__file__).parents[...]`` assumptions in moved modules.
"""

from __future__ import annotations

import sys
from pathlib import Path


def src_root(start: str | Path | None = None) -> Path:
    """Return ``Code/DINOv3/src`` from any file below that tree."""
    if start is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = Path(start).resolve()

    current = start_path.parent if start_path.suffix else start_path
    for candidate in (current, *current.parents):
        if (
            (candidate / "evaluation").is_dir()
            and (candidate / "data").is_dir()
            and (candidate / "models").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Could not locate DINOv3 src root from {start_path}")


def dino_root(start: str | Path | None = None) -> Path:
    """Return ``Code/DINOv3``."""
    return src_root(start).parent


def project_root(start: str | Path | None = None) -> Path:
    """Return the research project root."""
    return src_root(start).parents[2]


def ensure_src_on_path(start: str | Path | None = None) -> Path:
    """Put ``Code/DINOv3/src`` on ``sys.path`` and return it."""
    root = src_root(start)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def ensure_external_dinov3_on_path(start: str | Path | None = None) -> Path:
    """Put ``Code/DINOv3/external/dinov3`` on ``sys.path`` and return it."""
    external = dino_root(start) / "external" / "dinov3"
    external_str = str(external)
    if external_str not in sys.path:
        sys.path.insert(0, external_str)
    return external

