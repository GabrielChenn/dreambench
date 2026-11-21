"""Task goals and deterministic scoring rules for simulation replays."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Callable


TASK = "put apple into the bowl"
TASK_ID = "apple-in-bowl"
SCORER_ID = "apple-in-bowl-v1"
SUCCESS_CONDITION = "The released apple stays supported by the bowl and stable for at least 0.5 seconds."
STABLE_SECONDS = 0.50
MAX_STABLE_MOTION_M = 0.008
LIFT_M = 0.035
# Conservative center region for the official bowl's 83 mm outer radius and 58 mm rim.
INSIDE_RADIUS_M = 0.055
MIN_CENTER_HEIGHT_M = 0.020
MAX_CENTER_HEIGHT_M = 0.065


def in_bowl(row: dict) -> bool:
    objects = row["object_xyz_m"]
    apple, bowl = objects["apple"], objects["bowl"]
    contacts = row.get("apple_contacts", [])
    return (
        dist(apple[:2], bowl[:2]) <= INSIDE_RADIUS_M
        and MIN_CENTER_HEIGHT_M <= apple[2] - bowl[2] <= MAX_CENTER_HEIGHT_M
        and "bowl" in contacts
        and not any(name.startswith(("left_", "right_")) for name in contacts)
    )


def score_apple_in_bowl(rows: list[dict]) -> dict:
    if any("apple_contacts" not in row for row in rows):
        return {"outcome": "insufficient-evidence", "reason": "Recording predates apple contact telemetry.", "evidence_sim_s": None}

    # Use the final stable interval, so an apple later knocked out does not pass.
    final = rows[-1]
    final_apple = final["object_xyz_m"]["apple"]
    start = final["sim_seconds"]
    for row in reversed(rows):
        apple = row["object_xyz_m"]["apple"]
        if not in_bowl(row) or dist(apple, final_apple) > MAX_STABLE_MOTION_M:
            break
        start = row["sim_seconds"]
    if final["sim_seconds"] - start >= STABLE_SECONDS - 1e-6:
        return {"outcome": "success", "reason": "Apple finished supported by the bowl, clear of the arms, and stable.",
                "evidence_sim_s": round(start + STABLE_SECONDS, 3)}

    initial_height = rows[0]["object_xyz_m"]["apple"][2]
    peak = max(rows, key=lambda row: row["object_xyz_m"]["apple"][2])
    if peak["object_xyz_m"]["apple"][2] - initial_height < LIFT_M:
        # Report an observable event, not a claim about the model's internal perception.
        touched = next((row for row in rows if any(
            name.startswith(("left_", "right_")) for name in row["apple_contacts"]
        )), peak)
        return {"outcome": "task-failure", "reason": "Apple never lifted at least 35 mm above its initial height.",
                "evidence_sim_s": touched["sim_seconds"]}
    return {"outcome": "task-failure", "reason": "Apple lifted but did not finish stable and released in the bowl.",
            "evidence_sim_s": final["sim_seconds"]}


@dataclass(frozen=True)
class Task:
    scorer_id: str
    success_condition: str
    score: Callable[[list[dict]], dict]


TASKS = {TASK_ID: Task(SCORER_ID, SUCCESS_CONDITION, score_apple_in_bowl)}


def task_id(metadata: dict) -> str | None:
    if "task_id" in metadata:
        return metadata["task_id"]
    return TASK_ID if str(metadata.get("task", "")).strip().lower() == TASK else None


def score(rows: list[dict], metadata: dict) -> dict:
    if (metadata.get("dry_run") or metadata.get("status") != "finished"
            or metadata.get("recorder_exit_code", 0) != 0 or not rows):
        return {"outcome": "invalid-run", "reason": "No completed task capture to score.",
                "evidence_sim_s": None}
    task = TASKS.get(task_id(metadata))
    if task is None or metadata.get("scorer", task.scorer_id) != task.scorer_id:
        return {"outcome": "unsupported-task", "reason": "No matching scoring rule is registered.",
                "evidence_sim_s": None}
    return task.score(rows)
