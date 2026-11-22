"""Random scene layouts (every movable object moved) and the generalization table built from them.

An experiment may ask for random starts: `"random_starts": {"count": 5, "max_offset_m": 0.04, "seed": 1234}`.
Each start is a layout: every movable object of the scene (a body with a free joint: the fruit and the
water bottle) gets its own offset, drawn uniformly in the square ±max_offset_m along the MuJoCo world x
and y axes from the seed, applied on top of the scene layout after every reset. The bowl and plate are
fixed bodies of the scene and do not move. Each start is run as its own condition with the same
baseline (and candidate), so the report shows in which layouts the policy still succeeds.

Before any Dream-1 call, `layout_conflict()` rejects a layout where a moved object overlaps another
body (fruit, bowl, plate, an arm) or is no longer above the table; the sampler then draws again. The
same seed, scene and objects give the same layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import random
from typing import Callable

MAX_OFFSET_M = 0.10
MIN_OFFSET_M = 0.005
DEFAULT_OFFSET_M = 0.04
MAX_OFFSETS = 9
MAX_SEED = 2**31 - 1
DRAWS_PER_START = 50  # give up after this many rejected draws per requested start
TABLE_GEOM = "table"


@dataclass(frozen=True)
class ObjectOffset:
    object: str
    dx_m: float
    dy_m: float

    @property
    def label(self) -> str:
        return f"{self.object} Δx {self.dx_m * 100:+.1f} cm, Δy {self.dy_m * 100:+.1f} cm"

    @property
    def short(self) -> str:
        return f"{self.dx_m * 100:+.1f}, {self.dy_m * 100:+.1f}"

    def as_dict(self) -> dict:
        return {"object": self.object, "dx_m": self.dx_m, "dy_m": self.dy_m}

    @classmethod
    def from_dict(cls, data: object) -> "ObjectOffset | None":
        if not isinstance(data, dict):
            return None
        try:
            return cls(str(data["object"]), float(data["dx_m"]), float(data["dy_m"]))
        except (KeyError, TypeError, ValueError):
            return None


Layout = tuple[ObjectOffset, ...]


def layout_dicts(layout: Layout) -> list[dict]:
    return [offset.as_dict() for offset in layout]


def layout_from(data: object) -> Layout | None:
    """A layout from its JSON list, or None when it is missing or malformed."""
    if not isinstance(data, list) or not data:
        return None
    offsets = tuple(ObjectOffset.from_dict(item) for item in data)
    return None if any(offset is None for offset in offsets) else offsets


def validate_random_starts(raw: object) -> dict:
    """Return {"count", "max_offset_m", "seed"}, or raise ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("Describe random starts as {count, max_offset_m, seed}.")
    count, limit, seed = raw.get("count"), raw.get("max_offset_m", DEFAULT_OFFSET_M), raw.get("seed")
    if type(count) is not int or not 1 <= count <= MAX_OFFSETS:
        raise ValueError(f"Use 1 to {MAX_OFFSETS} random layouts.")
    if type(limit) not in (int, float) or not isfinite(limit) or not MIN_OFFSET_M <= limit <= MAX_OFFSET_M:
        raise ValueError(f"Keep the random start range from ±{MIN_OFFSET_M * 100:g} to ±{MAX_OFFSET_M * 100:g} cm.")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"Use an integer random seed from 0 to {MAX_SEED}.")
    if set(raw) - {"count", "max_offset_m", "seed"}:
        raise ValueError("Random starts take only count, max_offset_m and seed.")
    return {"count": count, "max_offset_m": round(float(limit), 4), "seed": seed}


def sample_layouts(spec: dict, objects: tuple[str, ...],
                   conflict: Callable[[Layout], "str | None"] | None = None) -> tuple[list[Layout], list[dict]]:
    """Draw spec["count"] layouts from spec["seed"]: one offset per object, in the given order. Redraw any
    layout that conflict() rejects. Returns the accepted layouts and the rejected draws with their reasons.
    Raises ValueError when too many draws are rejected (the range leaves no free space)."""
    if not objects:
        raise ValueError("The scene has no movable objects to randomize.")
    rng = random.Random(spec["seed"])
    limit = spec["max_offset_m"]
    accepted: list[Layout] = []
    rejected: list[dict] = []
    for _ in range(spec["count"] * DRAWS_PER_START):
        if len(accepted) == spec["count"]:
            break
        layout = tuple(ObjectOffset(name, round(rng.uniform(-limit, limit), 3), round(rng.uniform(-limit, limit), 3))
                       for name in objects)
        if layout in accepted:
            continue
        reason = conflict(layout) if conflict is not None else None
        if reason:
            rejected.append({"offsets": layout_dicts(layout), "reason": reason})
        else:
            accepted.append(layout)
    if len(accepted) < spec["count"]:
        raise ValueError(f"Found only {len(accepted)} of {spec['count']} usable layouts within ±{limit * 100:g} cm. "
                         "Use a smaller range or fewer starts.")
    return accepted, rejected


def free_qpos_address(model, body: str) -> int:
    """The qpos index of a body's free joint (x, y, z, then the quaternion)."""
    body_id = model.body(body).id
    if model.body_jntnum[body_id] < 1:
        raise ValueError(f"{body!r} has no free joint to move.")
    return int(model.jnt_qposadr[model.body_jntadr[body_id]])


def apply_layout(model, data, layout: Layout) -> None:
    import mujoco

    for offset in layout:
        address = free_qpos_address(model, offset.object)
        data.qpos[address] += offset.dx_m
        data.qpos[address + 1] += offset.dy_m
    mujoco.mj_forward(model, data)


def start_conflict(model, data, body: str) -> str | None:
    """Why the body's current start is unusable, or None. Call after mj_forward."""
    body_id = model.body(body).id
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist >= 0:
            continue
        bodies = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if body_id not in bodies:
            continue
        other = bodies[1] if bodies[0] == body_id else bodies[0]
        if other == 0:  # the table and other world geoms
            continue
        other_name = model.body(int(model.body_rootid[other])).name or model.body(other).name
        return f"the {body} starts inside {other_name}"
    table = model.geom(TABLE_GEOM)
    x, y = data.xpos[body_id][:2]
    (cx, cy), (sx, sy) = table.pos[:2], table.size[:2]
    if not (cx - sx <= x <= cx + sx and cy - sy <= y <= cy + sy):
        return f"the {body} starts off the table"
    return None


def layout_conflict(model, data, layout: Layout) -> str | None:
    """Why a just-applied layout is unusable (the first moved object that overlaps or left the table)."""
    for offset in layout:
        if reason := start_conflict(model, data, offset.object):
            return reason
    return None


def condition_result(pair: dict) -> dict:
    """One row of the generalization table: the condition and each arm's verdict."""
    row = {"scene_id": pair.get("scene_id"), "object_offsets": pair.get("object_offsets"),
           "folder": pair.get("folder")}
    for label in ("baseline", "candidate"):
        trial = pair.get(label)
        if isinstance(trial, dict):
            row[label] = {"outcome": trial.get("outcome"), "verdict": trial.get("verdict"),
                          "replay": trial.get("replay"), "evidence_sim_s": trial.get("evidence_sim_s")}
    return row


def generalization(pairs: list[dict]) -> dict:
    """Verdict counts per arm over every randomized layout. Runs without a valid score are counted as
    not scored, never as passes or failures."""
    rows = [condition_result(pair) for pair in pairs]
    totals = {}
    for label in ("baseline", "candidate"):
        trials = [row[label] for row in rows if label in row]
        if not trials:
            continue
        scored = [trial for trial in trials if trial["outcome"] in ("success", "task-failure")]
        passed = sum(trial["outcome"] == "success" for trial in scored)
        totals[label] = {"passed": passed, "failed": len(scored) - passed,
                         "scored": len(scored), "conditions": len(trials)}
    return {"rows": rows, "totals": totals}
