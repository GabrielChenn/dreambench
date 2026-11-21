"""Capture and reproduce a single YAM simulation control step without Dream-1."""

import hashlib

import mujoco
import numpy as np

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
STATE_SCHEMA = "mujoco-integration-v1"


def model_digest(model) -> str:
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    return hashlib.sha256(binary).hexdigest()


def integration_state(model, data) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    return state


def reproduce_step(env, row: dict) -> dict:
    snapshot = row.get("physics_before_step")
    if not isinstance(snapshot, dict) or snapshot.get("schema") != STATE_SCHEMA:
        raise ValueError("This frame has no pre-step physics snapshot. Record a new run to reproduce it.")
    if snapshot.get("mujoco_version") != mujoco.__version__ or snapshot.get("model_sha256") != model_digest(env.model):
        raise ValueError("Use the same MuJoCo version and YAM model as the original recording.")
    state = np.asarray(snapshot["values"], dtype=float)
    if state.shape != (mujoco.mj_stateSize(env.model, STATE),) or not np.isfinite(state).all():
        raise ValueError("Invalid MuJoCo state vector.")
    commanded = row["commanded_joint_rad_and_gripper"]
    mujoco.mj_setState(env.model, env.data, state, STATE)
    mujoco.mj_forward(env.model, env.data)
    # Restore integration components again in case the forward pass updated solver state.
    mujoco.mj_setState(env.model, env.data, state, STATE)
    actual = env.step(commanded["left"], commanded["right"])
    measured = row["measured_joint_rad_and_gripper"]
    joint_error = max(float(np.max(np.abs(actual[side][:6] - np.asarray(measured[side][:6]))))
                      for side in ("left", "right"))
    gripper_error = max(abs(float(actual[side][6]) - measured[side][6]) for side in ("left", "right"))
    object_error = max(float(np.max(np.abs(env.data.body(name).xpos - xyz)))
                       for name, xyz in row["object_xyz_m"].items())
    tolerance = 1e-8
    return {
        "frame": row["frame"], "sim_seconds": row["sim_seconds"],
        "matches_recording": max(joint_error, gripper_error, object_error) <= tolerance,
        "absolute_tolerance": tolerance,
        "max_joint_error_rad": joint_error, "max_gripper_error": gripper_error,
        "max_object_position_error_m": object_error,
        "measured_joint_rad_and_gripper": {side: values.tolist() for side, values in actual.items()},
    }
