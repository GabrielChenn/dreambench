"""Name how a trial failed, from the scorer's diagnosis of the judged object, and count modes across trials.

The mode is read from diagnose() (scoring.py), which uses simulator ground truth: contacts and the
object's trajectory. It describes what was observed, not why the model chose it.
"""

from __future__ import annotations

from collections.abc import Iterable

# (key, label, what it means) in display order: success first, then the task stages in order.
MODES = (
    ("successed", "Successed", "Placed, released and stable."),
    ("never-reached", "Never reached the object", "The arms never touched the judged object."),
    ("knocked-away", "Knocked away", "The object moved without arm contact, e.g. hit by another object."),
    ("grasp-failed", "Grasp failed", "The arms touched the object but never lifted it 35 mm."),
    ("still-holding", "Still holding at the end", "Lifted, and still in the gripper when the run ended."),
    ("dropped-or-misplaced", "Dropped or placed off target", "Lifted, then released away from the target."),
    ("not-released", "Not released cleanly", "Ended touching the target but also an arm or the table."),
    ("unsettled", "On target, not settled", "Ended on the target but not stable long enough."),
    ("not-scored", "Not scored", "Not run, invalid start, run error, or no usable recording."),
    ("unknown", "Unknown", "Scored, but the recording has no diagnosis of the object."),
)
LABELS = {key: label for key, label, _ in MODES}
SCORED = ("success", "task-failure")


def holds(contacts: Iterable[str] | None) -> bool:
    return any(name.startswith(("left_", "right_")) for name in contacts or ())


def failure_mode(trial: dict) -> str:
    """The mode key for one trial result (assess() or a not-run placeholder)."""
    if trial.get("outcome") not in SCORED:
        return "not-scored"
    if trial.get("outcome") == "success":
        return "successed"
    diagnosis = trial.get("diagnostics")
    stage = diagnosis.get("stage") if isinstance(diagnosis, dict) else None
    if stage == "lifted-not-placed":
        return "still-holding" if holds(diagnosis.get("final_contacts")) else "dropped-or-misplaced"
    return {"untouched": "never-reached", "moved-without-arm-contact": "knocked-away",
            "touched-not-lifted": "grasp-failed", "touching-target": "not-released",
            "on-target": "unsettled"}.get(stage, "unknown")


def mode_distribution(pairs: list[dict]) -> dict:
    """Per arm: how many trials fell in each mode, and which conditions (1-based) they were."""
    result: dict = {}
    for label in ("baseline", "candidate"):
        trials = [(index, pair[label]) for index, pair in enumerate(pairs, 1) if isinstance(pair.get(label), dict)]
        if not trials:
            continue
        modes: dict = {}
        for index, trial in trials:
            entry = modes.setdefault(failure_mode(trial), {"count": 0, "conditions": []})
            entry["count"] += 1
            entry["conditions"].append(pair_number(pairs[index - 1], index))
        result[label] = {key: modes[key] for key, _, _ in MODES if key in modes}
    return result


def pair_number(pair: dict, index: int) -> int:
    start = pair.get("start")
    return start if type(start) is int else index
