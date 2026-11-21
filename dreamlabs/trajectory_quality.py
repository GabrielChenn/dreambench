"""Descriptive joint-space metrics from synchronized simulation telemetry."""

from collections import deque
from math import isclose, isfinite, sqrt

ARMS = ("left", "right")
MEASURED = "measured_joint_rad_and_gripper"
TARGET = "commanded_joint_rad_and_gripper"


def joint_vector(row: dict, key: str) -> tuple:
    state = row.get(key)
    if not isinstance(state, dict):
        return ()
    values = [state.get(side) for side in ARMS]
    if any(not isinstance(arm, (list, tuple)) or len(arm) != 7 for arm in values):
        return ()
    joints = tuple(value for arm in values for value in arm[:6])
    if not all(type(value) in (float, int) and isfinite(value) for value in joints):
        return ()
    return joints


def trajectory_quality(rows: list[dict]) -> dict:
    errors_sq = []
    jerks = []
    window: deque = deque(maxlen=4)
    for row in rows:
        measured = joint_vector(row, MEASURED)
        target = joint_vector(row, TARGET)
        if measured and target:
            errors_sq.extend((actual - commanded) ** 2 for actual, commanded in zip(measured, target))

        timestamp = row.get("sim_seconds")
        if not measured or type(timestamp) not in (int, float) or not isfinite(timestamp):
            window.clear()
            continue
        if window and not isclose(timestamp - window[-1][0], 0.05, abs_tol=1e-6):
            window.clear()
        window.append((timestamp, measured))
        if len(window) == 4:
            first, second, third, fourth = (sample[1] for sample in window)
            jerks.extend(abs(d - 3 * c + 3 * b - a) / (0.05 ** 3)
                         for a, b, c, d in zip(first, second, third, fourth))

    jerks.sort()
    # Nearest-rank percentile across the 12 joint measurements and aligned time windows.
    rank = (95 * len(jerks) + 99) // 100
    return {
        "method": "yam-joint-telemetry-v1",
        "tracking_rmse_rad": sqrt(sum(errors_sq) / len(errors_sq)) if errors_sq else None,
        "joint_jerk_p95_rad_s3": jerks[rank - 1] if jerks else None,
        "tracking_frames": len(errors_sq) // 12,
        "jerk_windows": len(jerks) // 12,
    }
