"""Render 3D reviews of recordings for the Web workbench, one request per stdin line.

    {"op": "summary", "recording": "/abs/replay-x"}
    {"op": "render", "recording": "/abs/replay-x", "frame": 286, "out": "/tmp/frame.jpg",
     "camera": {"lookat": [x, y, z], "distance": 1.2, "azimuth": 90, "elevation": -25},
     "contacts": true, "other": false, "trails": true}

Each answer is one line, "@@review " followed by JSON. It makes no API calls and never moves the
live simulator: it restores recorded states into its own copy of the YAM scene.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PREFIX = "@@review "


def answer(**payload) -> None:
    sys.stdout.write(PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def camera(request: dict, defaults: dict) -> dict:
    """Validate a requested camera, falling back to the desktop viewer's angle."""
    raw = request.get("camera") or {}
    result = dict(defaults)
    for key, low, high in (("distance", 0.15, 6.0), ("azimuth", -720.0, 720.0), ("elevation", -89.0, 89.0)):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high:
            result[key] = float(value)
    lookat = raw.get("lookat")
    if (isinstance(lookat, list) and len(lookat) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 5 for v in lookat)):
        result["lookat"] = [float(v) for v in lookat]
    return result


def main() -> None:
    from dreamlabs.review_scene import ReviewError, ReviewScene
    from dreamlabs.trial_recorder import load_upstream

    _, yam_env = load_upstream()
    defaults = {"lookat": [float(v) for v in yam_env.VIEWER_LOOKAT], "distance": float(yam_env.VIEWER_DISTANCE),
                "azimuth": float(yam_env.VIEWER_AZIMUTH), "elevation": float(yam_env.VIEWER_ELEVATION)}
    scene = ReviewScene(yam_env.build_xml, defaults)
    answer(ok=True, ready=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            path = Path(request["recording"])
            if request.get("op") == "summary":
                answer(ok=True, summary=scene.summary(path))
            elif request.get("op") == "render":
                image = scene.render(path, int(request["frame"]), camera(request, defaults),
                                     contacts=request.get("contacts") is not False,
                                     other=request.get("other") is True, trails=request.get("trails") is not False)
                Path(request["out"]).write_bytes(image)
                answer(ok=True)
            else:
                answer(ok=False, error="Unknown review request.")
        except (ReviewError, KeyError, TypeError, ValueError, OSError) as error:
            answer(ok=False, error=str(error) or type(error).__name__)


if __name__ == "__main__":
    main()
