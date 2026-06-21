"""Shared parsing for 2D and pseudo-2.5D channel modes."""

from __future__ import annotations

import re

ALLOWED_25D_CHANNELS = (3, 5, 7, 9)
VALID_MODE_HELP = "'2d', '2d_1ch', or '2.5d_{N}ch' with N in {3, 5, 7, 9}"

_MODE_RE = re.compile(r"^2\.5d_(\d+)ch$")


def mode_channel_count(mode: str) -> int:
    """Return the input-channel count implied by a data mode."""
    if mode == "2d_1ch":
        return 1
    if mode == "2d":
        return 3

    match = _MODE_RE.fullmatch(mode)
    if match is None:
        raise ValueError(f"Invalid input mode {mode!r}; expected {VALID_MODE_HELP}")

    channels = int(match.group(1))
    if channels not in ALLOWED_25D_CHANNELS:
        allowed = ", ".join(str(ch) for ch in ALLOWED_25D_CHANNELS)
        raise ValueError(
            f"Invalid 2.5D channel count {channels}; expected one of {{{allowed}}}"
        )
    return channels


def is_25d_mode(mode: str) -> bool:
    """Return whether mode names a supported neighboring-slice input."""
    if _MODE_RE.fullmatch(mode) is None:
        return False
    mode_channel_count(mode)
    return True


def make_offsets(mode: str, neighbor_stride: int) -> list[int]:
    """Return per-channel slice offsets relative to the central slice t."""
    if neighbor_stride < 1:
        raise ValueError(f"neighbor_stride must be >= 1, got {neighbor_stride}")

    if mode == "2d_1ch":
        return [0]

    if mode == "2d":
        return [0, 0, 0]

    channels = mode_channel_count(mode)
    radius = channels // 2
    return [offset * neighbor_stride for offset in range(-radius, radius + 1)]
