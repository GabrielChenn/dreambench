"""Task goals and deterministic scoring rules for simulation replays.

Every movable object of the YAM tabletop scene (apple, banana, orange, water bottle) has a scorer for
each fixed container (bowl, plate): eight calibrated tasks, listed in TASKS. Any other
"<object>-in|on-<support>" pair is judged by the generic placement rule (generic-placement-v1).
Every score also carries diagnostics: whether the arms touched the object, how high it was lifted,
where it ended up and what it touched, so a run gets feedback even when it cannot be judged.

Ported from the dreambench_bk judging work. apple-in-bowl-v1 is unchanged, so earlier apple results
stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
import re


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

# Scene bodies, from dream_examples/yam_mujoco/yam_env.py. Only the objects have free joints; the
# bowl and plate are fixed, so they are always the destination. The table is a world geom, so
# contacts with it report the body name "world".
OBJECTS = ("apple", "banana", "orange", "water_bottle")
CONTAINERS = ("bowl", "plate")
TABLE_BODY = "world"
ARM_PREFIXES = ("left_", "right_")
# Outer radius of each container's support surface: bowl lathe profile 83 mm; plate 69 mm disc with
# a 5 mm rim capsule at 68 mm.
CONTAINER_RADIUS_M = {"bowl": 0.083, "plate": 0.074}
# How far an object's body origin can sit from its geometric center. The banana's origin is at the
# end of its scan: settled inside the bowl touching nothing else it measured up to 52 mm off-center.
ORIGIN_OFFSET_M = {"apple": 0.0, "banana": 0.060, "orange": 0.0, "water_bottle": 0.0}
PHRASE = {"apple": "apple", "banana": "banana", "orange": "orange", "water_bottle": "water bottle"}
RELATION = {"bowl": ("in", "into", "in"), "plate": ("on", "on", "on")}
DEFAULT_RELATION = ("on", "on", "on")
GENERIC_SCORER = "generic-placement-v1"
CALIBRATED, GENERIC = "calibrated", "generic"
TASK_ID_PATTERN = re.compile(r"^(?P<object>[a-z0-9]+(?:-[a-z0-9]+)*?)-(?P<relation>in|on)-(?P<target>[a-z0-9_-]+)$")

# Judged verdicts, as in dreambench_bk/judge_run.py.
SUCCESSED, FAILED, TIMEOUT_STOPPED = "successed", "failed", "timeout stopped"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    object: str
    target: str
    scorer: str
    prompt: str
    success_condition: str
    tier: str = CALIBRATED  # CALIBRATED: thresholds measured in MuJoCo; GENERIC: contact rule only


def phrase(name: str) -> str:
    return PHRASE.get(name, name.replace("_", " "))


def make_task(obj: str, target: str, tier: str = CALIBRATED) -> TaskSpec:
    short, verb, where = RELATION.get(target, DEFAULT_RELATION)
    task_id = f"{obj.replace('_', '-')}-{short}-{target.replace('_', '-')}"
    if (obj, target) == ("apple", "bowl") and tier == CALIBRATED:
        return TaskSpec(TASK_ID, obj, target, SCORER_ID, TASK, SUCCESS_CONDITION)
    scorer = f"{task_id}-v1" if tier == CALIBRATED else GENERIC_SCORER
    return TaskSpec(task_id, obj, target, scorer, f"put {phrase(obj)} {verb} the {phrase(target)}",
                    f"The released {phrase(obj)} rests {where} the {phrase(target)}, touching neither the table "
                    f"nor the arms, and stays stable for at least {STABLE_SECONDS:g} seconds.", tier)


TASKS: dict[str, TaskSpec] = {spec.task_id: spec for spec in (
    make_task(obj, target) for target in CONTAINERS for obj in OBJECTS)}
TASK_BY_PAIR: dict[tuple[str, str], TaskSpec] = {(spec.object, spec.target): spec for spec in TASKS.values()}
TASK_BY_PROMPT: dict[str, TaskSpec] = {spec.prompt: spec for spec in TASKS.values()}


def task_for(obj: str, target: str) -> TaskSpec:
    """The calibrated task for this pair, or a generic one."""
    return TASK_BY_PAIR.get((obj, target)) or make_task(obj, target, GENERIC)


def resolve_task(task: str | None, metadata: dict | None = None) -> TaskSpec | None:
    """Find the scorer for a task ID: calibrated if registered, else the generic placement rule.

    Explicit "object" and "target" in the run metadata win over parsing the ID, because object names
    may contain underscores that the ID spells as hyphens.
    """
    if task in TASKS:
        return TASKS[task]
    metadata = metadata or {}
    if isinstance(metadata.get("object"), str) and isinstance(metadata.get("target"), str):
        return task_for(metadata["object"], metadata["target"])
    match = TASK_ID_PATTERN.match(task or "") if isinstance(task, str) else None
    if not match:
        return None
    return task_for(match["object"].replace("-", "_"), match["target"].replace("-", "_"))


def task_id(metadata: dict) -> str | None:
    if "task_id" in metadata:
        return metadata["task_id"]
    spec = TASK_BY_PROMPT.get(str(metadata.get("task", "")).strip().lower())
    return spec.task_id if spec else None


def contacts_key(obj: str) -> str:
    return f"{obj}_contacts"


def touches_arm(contacts: list[str]) -> bool:
    return any(name.startswith(ARM_PREFIXES) for name in contacts)


def in_bowl(row: dict) -> bool:
    """apple-in-bowl-v1: unchanged, so earlier apple results stay comparable."""
    objects = row["object_xyz_m"]
    apple, bowl = objects["apple"], objects["bowl"]
    contacts = row.get("apple_contacts", [])
    return (
        dist(apple[:2], bowl[:2]) <= INSIDE_RADIUS_M
        and MIN_CENTER_HEIGHT_M <= apple[2] - bowl[2] <= MAX_CENTER_HEIGHT_M
        and "bowl" in contacts
        and not touches_arm(contacts)
    )


def placed(row: dict, spec: TaskSpec) -> bool:
    """True when the object is supported by the target alone: the table and arms are clear."""
    if spec.task_id == TASK_ID:
        return in_bowl(row)
    objects = row["object_xyz_m"]
    item, target = objects[spec.object], objects[spec.target]
    contacts = row.get(contacts_key(spec.object), [])
    radius = CONTAINER_RADIUS_M.get(spec.target)
    # Unmeasured supports have no known radius: rely on the contact rule alone.
    near = radius is None or dist(item[:2], target[:2]) <= radius + ORIGIN_OFFSET_M.get(spec.object, 0.0)
    return (
        spec.target in contacts
        and TABLE_BODY not in contacts
        and not touches_arm(contacts)
        and near
        and item[2] > target[2]
    )


def stable_since(rows: list[dict], spec: TaskSpec) -> float | None:
    """Start of the final interval in which the object stays placed and still, or None."""
    final_xyz = rows[-1]["object_xyz_m"][spec.object]
    start = None
    for row in reversed(rows):
        if not placed(row, spec) or dist(row["object_xyz_m"][spec.object], final_xyz) > MAX_STABLE_MOTION_M:
            break
        start = row["sim_seconds"]
    return start


def diagnose(rows: list[dict], obj: str, target: str | None) -> dict | None:
    """Describe what happened to the object, whether or not the task can be judged.

    Positions are body origins in the world frame; distances are in mm and times in sim seconds.
    Returns None when the recording has no position for the object.
    """
    if not rows or any(obj not in row.get("object_xyz_m", {}) for row in rows):
        return None
    key = contacts_key(obj)
    has_contacts = all(key in row for row in rows)
    xyz = [row["object_xyz_m"][obj] for row in rows]
    initial, final = xyz[0], xyz[-1]
    peak = max(range(len(rows)), key=lambda i: xyz[i][2])
    lift_mm = round((xyz[peak][2] - initial[2]) * 1000, 1)
    arm_times = [row["sim_seconds"] for row in rows if has_contacts and touches_arm(row[key])]
    final_contacts = sorted(rows[-1][key]) if has_contacts else None
    info: dict = {
        "object": obj, "target": target, "contacts_recorded": has_contacts,
        "first_arm_contact_sim_s": arm_times[0] if arm_times else None,
        "last_arm_contact_sim_s": arm_times[-1] if arm_times else None,
        "max_lift_mm": lift_mm, "max_lift_sim_s": rows[peak]["sim_seconds"],
        "displacement_mm": round(dist(initial, final) * 1000, 1),
        "final_xyz_m": [round(v, 4) for v in final], "final_contacts": final_contacts,
        "ended_on_table": None if final_contacts is None else TABLE_BODY in final_contacts,
        "final_target_xy_distance_mm": None, "final_height_above_target_mm": None,
    }
    in_target = False
    target_xyz = rows[-1]["object_xyz_m"].get(target) if target else None
    if target_xyz is not None:
        info["final_target_xy_distance_mm"] = round(dist(final[:2], target_xyz[:2]) * 1000, 1)
        info["final_height_above_target_mm"] = round((final[2] - target_xyz[2]) * 1000, 1)
        in_target = final_contacts is not None and target in final_contacts
    lifted = lift_mm >= LIFT_M * 1000

    if not has_contacts:
        stage = "lifted" if lifted else "not-lifted"
    elif in_target and not touches_arm(final_contacts) and TABLE_BODY not in final_contacts:
        stage = "on-target"
    elif in_target:
        stage = "touching-target"
    elif lifted:
        stage = "lifted-not-placed"
    elif arm_times:
        stage = "touched-not-lifted"
    elif info["displacement_mm"] > 10:
        stage = "moved-without-arm-contact"
    else:
        stage = "untouched"
    info["stage"] = stage
    info["summary"] = summarize(info, phrase(obj), phrase(target) if target else "target")
    return info


def summarize(info: dict, name: str, where: str) -> str:
    """One sentence for a diagnose() result. Only the chosen stage's fields are read."""
    stage, lift = info["stage"], info["max_lift_mm"]
    if stage == "untouched":
        return f"The arms never touched the {name}."
    if stage == "moved-without-arm-contact":
        return f"The {name} moved {info['displacement_mm']:g} mm without arm contact (knocked by another object?)."
    if stage == "touched-not-lifted":
        return (f"The arms touched the {name} at {info['first_arm_contact_sim_s']:.2f} s but lifted it only "
                f"{lift:g} mm (35 mm needed): the grasp did not hold.")
    if stage == "lifted-not-placed":
        distance = info["final_target_xy_distance_mm"]
        return (f"The {name} was lifted {lift:g} mm at {info['max_lift_sim_s']:.2f} s but ended off the {where}"
                + (f", {distance:g} mm from its center" if distance is not None else "")
                + (" on the table." if info["ended_on_table"] else "."))
    if stage == "touching-target":
        other = "the table" if TABLE_BODY in info["final_contacts"] else "an arm"
        return f"The {name} ended touching the {where} but also {other}: not released cleanly onto it."
    if stage == "on-target":
        prep = RELATION.get(info["target"], DEFAULT_RELATION)[2]
        return f"The {name} ended resting {prep} the {where} with the arms and table clear."
    return f"The {name} was lifted {lift:g} mm (contacts not recorded)."


def score(rows: list[dict], metadata: dict) -> dict:
    result = score_outcome(rows, metadata)
    spec = resolve_task(task_id(metadata), metadata)
    result["tier"] = spec.tier if spec else None
    result["diagnostics"] = diagnose(rows, spec.object, spec.target) if spec and rows else None
    return result


def score_outcome(rows: list[dict], metadata: dict) -> dict:
    if (metadata.get("dry_run") or metadata.get("status") != "finished"
            or metadata.get("recorder_exit_code", 0) != 0 or not rows):
        return {"outcome": "invalid-run", "reason": "No completed task capture to score.", "evidence_sim_s": None}
    spec = resolve_task(task_id(metadata), metadata)
    if spec is None or metadata.get("scorer", spec.scorer) != spec.scorer:
        return {"outcome": "unsupported-task", "reason": f"No scorer is registered for task {task_id(metadata)!r}.",
                "evidence_sim_s": None}
    obj, key, name = spec.object, contacts_key(spec.object), phrase(spec.object).capitalize()
    if any(obj not in row.get("object_xyz_m", {}) or spec.target not in row["object_xyz_m"] for row in rows):
        return {"outcome": "unsupported-task",
                "reason": f"The recording has no {phrase(obj)} or {phrase(spec.target)}: it is not in this scene.",
                "evidence_sim_s": None}
    if any(key not in row for row in rows):
        return {"outcome": "insufficient-evidence", "reason": f"Recording predates {phrase(obj)} contact telemetry.",
                "evidence_sim_s": None}

    # Use the final stable interval, so an object later knocked out does not pass.
    final = rows[-1]
    start = stable_since(rows, spec)
    if start is not None and final["sim_seconds"] - start >= STABLE_SECONDS - 1e-6:
        return {"outcome": "success",
                "reason": f"{name} finished supported by the {phrase(spec.target)}, clear of the arms, and stable.",
                "evidence_sim_s": round(start + STABLE_SECONDS, 3)}

    initial_height = rows[0]["object_xyz_m"][obj][2]
    peak = max(rows, key=lambda row: row["object_xyz_m"][obj][2])
    if peak["object_xyz_m"][obj][2] - initial_height < LIFT_M:
        # Report an observable event, not a claim about the model's internal perception.
        touched = next((row for row in rows if touches_arm(row[key])), peak)
        return {"outcome": "task-failure", "reason": f"{name} never lifted at least 35 mm above its initial height.",
                "evidence_sim_s": touched["sim_seconds"]}
    return {"outcome": "task-failure",
            "reason": f"{name} lifted but did not finish stable and released "
                      f"{RELATION.get(spec.target, DEFAULT_RELATION)[2]} the {phrase(spec.target)}.",
            "evidence_sim_s": final["sim_seconds"]}


def verdict(outcome: str | None, stop_reason: str | None) -> str:
    """Map a scored outcome to successed / failed / timeout stopped, as dreambench_bk/judge_run.py does.

    A run stopped because the arms stayed home is "timeout stopped" unless the object was placed first.
    Every other run that did not succeed (time limit, task failure, error, operator stop) is "failed".
    """
    if outcome == "success":
        return SUCCESSED
    if stop_reason == "home":
        return TIMEOUT_STOPPED
    return FAILED
