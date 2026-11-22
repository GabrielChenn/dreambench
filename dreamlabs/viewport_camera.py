"""The live viewport's free camera as sent by the configuration page: checked and clamped. No MuJoCo import."""

from __future__ import annotations

import math

ELEVATION_DEG = (-89.0, 89.0)
DISTANCE_M = (0.15, 6.0)
LOOKAT_M = 5.0  # |x|, |y|, |z| bound for the point the camera orbits, in world metres


def clean_camera(view: object) -> dict:
    """Return {azimuth, elevation, distance, lookat} in degrees and metres, or raise ValueError.

    Azimuth wraps to [0, 360); elevation, distance and the look-at point are clamped to the bounds above."""
    if not isinstance(view, dict):
        raise ValueError("Send the camera as an object.")
    numbers = [view.get(key) for key in ("azimuth", "elevation", "distance")]
    lookat = view.get("lookat")
    if not isinstance(lookat, list) or len(lookat) != 3:
        raise ValueError("Send lookat as three numbers.")
    numbers += lookat
    if not all(type(number) in (int, float) and math.isfinite(number) for number in numbers):
        raise ValueError("Camera values must be finite numbers.")
    azimuth, elevation, distance, *point = (float(number) for number in numbers)
    return {
        "azimuth": azimuth % 360.0,
        "elevation": min(max(elevation, ELEVATION_DEG[0]), ELEVATION_DEG[1]),
        "distance": min(max(distance, DISTANCE_M[0]), DISTANCE_M[1]),
        "lookat": [min(max(value, -LOOKAT_M), LOOKAT_M) for value in point],
    }
