"""Find which bodies of the simulated scene can be moved and which can hold them.

Nothing here is hard-coded per object: an object added to the MuJoCo scene is picked up the next time
the scene is built, so it gets telemetry, a generic scorer and diagnostics without new code.

Rules, applied to the compiled MuJoCo model:
- object:  a body with a free joint (it can be picked up and moved).
- support: a direct child of the world body with no joints anywhere in its subtree (so not a robot
           arm) and at least one colliding geom (so it can hold something). Camera mounts are left
           out by name, since they collide but are not destinations.
The table is a geom of the world body itself, so contacts with it report "world"; it is not listed as
a support.

When the simulator is not installed, load_inventory() falls back to the bodies measured in scoring
(apple, banana, orange, water bottle; bowl, plate). Ported from dreambench_bk/scene_inventory.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import sys

from .scoring import CONTAINERS, OBJECTS

MJ_JOINT_FREE = 0  # mujoco.mjtJoint.mjJNT_FREE
NOT_A_SUPPORT = ("camera",)


@dataclass(frozen=True)
class Inventory:
    objects: tuple[str, ...]
    supports: tuple[str, ...]
    source: str  # "simulator" or "fallback"


FALLBACK = Inventory(OBJECTS, CONTAINERS, "fallback")


def discover(model) -> Inventory:
    """Read objects and supports from a compiled mujoco.MjModel."""
    objects = tuple(sorted(model.body(int(model.jnt_bodyid[j])).name
                           for j in range(model.njnt) if int(model.jnt_type[j]) == MJ_JOINT_FREE))
    joints_in_root: dict[int, int] = {}
    for body in range(model.nbody):
        root = int(model.body_rootid[body])
        joints_in_root[root] = joints_in_root.get(root, 0) + int(model.body_jntnum[body])
    colliding = {int(model.geom_bodyid[g]) for g in range(model.ngeom)
                 if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])}
    supports = tuple(sorted(
        model.body(body).name for body in range(1, model.nbody)
        if int(model.body_parentid[body]) == 0 and joints_in_root.get(body, 0) == 0 and body in colliding
        and not any(word in model.body(body).name for word in NOT_A_SUPPORT)))
    return Inventory(objects, supports, "simulator")


@lru_cache(maxsize=1)
def load_inventory() -> Inventory:
    """Build the YAM scene model (no rendering, no API call) and discover it, or use the fallback."""
    try:
        # find_spec raises, rather than returning None, when the parent package is missing.
        spec = importlib.util.find_spec("dream_examples.yam_mujoco")
    except ImportError:
        return FALLBACK
    if spec is None or not spec.submodule_search_locations:
        return FALLBACK
    try:
        import mujoco
        location = str(next(iter(spec.submodule_search_locations)))
        if location not in sys.path:
            sys.path.insert(0, location)
        from yam_env import build_xml
        return discover(mujoco.MjModel.from_xml_string(build_xml()))
    except Exception:  # noqa: BLE001 - any build failure means we cannot trust discovery
        return FALLBACK
