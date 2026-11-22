"""Whether offline YOLO can run here, and what happened to it in a recording. Imports nothing heavy."""

from __future__ import annotations

from importlib.machinery import PathFinder
import importlib.util
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT / ".venv" / "yolo-deps"
LOCAL_WEIGHTS = PROJECT / ".venv" / "yolo11n.pt"
DETECTIONS = "detections-yolo.jsonl"


def yolo_unavailable(weights: Path = LOCAL_WEIGHTS) -> str | None:
    """Why detect_replay.py cannot run, or None when its weights and Ultralytics are both present."""
    missing = []
    if not weights.is_file():
        missing.append(f"weights {weights}")
    found = PathFinder.find_spec("ultralytics", [str(LOCAL_DEPS)]) if LOCAL_DEPS.is_dir() else None
    if found is None and importlib.util.find_spec("ultralytics") is None:
        missing.append(f"the ultralytics package (in this Python or {LOCAL_DEPS})")
    return "YOLO is not installed: missing " + " and ".join(missing) + "." if missing else None


def read_detections(folder: Path) -> list[dict] | None:
    """The recording's YOLO detections, or None when it has none (or they cannot be read)."""
    path = folder / DETECTIONS
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return None


def yolo_note(folder: Path, detections: list[dict] | None) -> str:
    """One line for the camera views: are the boxes shown, and if not, why."""
    if detections is not None:
        ran = sum(1 for record in detections if record.get("ran"))
        return f"YOLO boxes from {DETECTIONS}: detection ran on {ran} of {len(detections)} frames."
    try:
        status = json.loads((folder / "run.json").read_text(encoding="utf-8")).get("yolo")
    except (OSError, ValueError, AttributeError):
        status = None
    if isinstance(status, dict) and status.get("status") == "failed":
        return f"YOLO was requested for this run but failed: {status.get('error') or 'no reason recorded'}"
    return "No YOLO detections for this recording (YOLO was not run)."
