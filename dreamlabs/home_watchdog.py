"""Stop a trial once the arms have left their initial pose, come back and stayed there.

Ported from dreambench_bk/judge_run.py. Only the counted joints of each arm are compared, joint2 and
joint3 (shoulder and elbow) by default: the wrist and base joints of an idle arm drift, so they never
settle back at home. Angles are in radians and times are simulated seconds from the trajectory rows.
Inference waits pause simulated time, so they do not count toward the dwell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIDES = ("left", "right")
DEFAULT_HOME_SECONDS = 5.0
DEFAULT_TOLERANCE_RAD = 0.25
DEFAULT_DEPART_RAD = 0.5
DEFAULT_JOINTS = (2, 3)


def joint_list(text: str) -> tuple[int, ...]:
    """Parse joints such as "2,3" or "1-6". Joints are numbered 1-6 within each arm."""
    joints: set[int] = set()
    for part in text.split(","):
        low, _, high = part.strip().partition("-")
        if not low.isdecimal() or (high and not high.isdecimal()):
            raise ValueError("Use joint numbers 1-6, e.g. 2,3 or 1-6.")
        joints.update(range(int(low), int(high or low) + 1))
    if not joints or not joints <= set(range(1, 7)):
        raise ValueError("Use joint numbers 1-6, e.g. 2,3 or 1-6.")
    return tuple(sorted(joints))


@dataclass
class HomeWatchdog:
    """Detect that the arms left the initial pose, came back near it, and stayed there."""

    home_seconds: float = DEFAULT_HOME_SECONDS
    tolerance_rad: float = DEFAULT_TOLERANCE_RAD
    depart_rad: float = DEFAULT_DEPART_RAD
    joints: tuple[int, ...] = DEFAULT_JOINTS
    home: list[float] | None = field(default=None, repr=False)
    departed_at: float | None = None
    near_since: float | None = None
    last_sim: float | None = None

    def __post_init__(self) -> None:
        if not (self.home_seconds > 0 and 0 < self.tolerance_rad < self.depart_rad):
            raise ValueError("Use positive home seconds and 0 < tolerance < depart radians.")

    def pose(self, row: dict) -> list[float] | None:
        measured = row.get("measured_joint_rad_and_gripper") or {}
        try:
            return [float(measured[side][j - 1]) for side in SIDES for j in self.joints]
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def observe(self, row: dict) -> None:
        pose = self.pose(row)
        if pose is None:
            return
        self.last_sim = float(row["sim_seconds"])
        if self.home is None:
            self.home = pose
            return
        deviation = max(abs(a - b) for a, b in zip(pose, self.home))
        if self.departed_at is None:
            if deviation > self.depart_rad:
                self.departed_at = self.last_sim
            return
        if deviation <= self.tolerance_rad:
            if self.near_since is None:
                self.near_since = self.last_sim
        else:
            self.near_since = None

    def should_stop(self) -> bool:
        return (self.near_since is not None and self.last_sim is not None
                and self.last_sim - self.near_since > self.home_seconds - 1e-9)

    def summary(self) -> dict:
        return {"joints": list(self.joints), "home_seconds": self.home_seconds,
                "tolerance_rad": self.tolerance_rad, "depart_rad": self.depart_rad,
                "departed_sim_s": self.departed_at, "home_since_sim_s": self.near_since,
                "last_sim_s": self.last_sim, "stop": self.should_stop()}
