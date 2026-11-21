"""Record one Dream YAM Sim trial: synchronized video, telemetry rows, run metadata and replay page.

Shared by record_replay.py (one process per trial) and sim_workspace.py (one persistent simulator).
Import it only in the Dream Labs environment: it needs MuJoCo, NumPy and PyAV.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Callable

import av
import mujoco
import numpy as np

from .contact_events import sample_contacts
from .frame_restore import STATE_SCHEMA, integration_state, model_digest
from .home_watchdog import HomeWatchdog
from .replay_page import CAMERAS, write_replay
from .scenarios import DEFAULT_SCENE_ID, SCENES, Scene
from .scene_inventory import discover
from .scoring import contacts_key

FPS = 20
VIEWPORT_PERIOD_S = 0.16


def body_contacts(model, data, name: str) -> list[str]:
    """Bodies touching the named body right now (penetrating or touching contacts only)."""
    body = model.body(name).id
    touching = set()
    for contact in data.contact[: data.ncon]:
        if contact.dist > 0:
            continue
        first, second = (int(model.geom_bodyid[g]) for g in contact.geom)
        if first == body:
            touching.add(model.body(second).name)
        elif second == body:
            touching.add(model.body(first).name)
    return sorted(touching)


class TrialRecorder:
    """Write one trial under `output`. Attach it to a recording env, run the policy, then close()."""

    def __init__(self, output: Path, model, *, watchdog: HomeWatchdog | None = None,
                 on_home: Callable[[], None] | None = None) -> None:
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.scene = discover(model)
        self.model_sha256 = model_digest(model)
        self.watchdog = watchdog
        self.on_home = on_home
        self.home_stop = False
        self.rows: list[dict] = []
        self.targets: dict | None = None
        self.before_step: dict | None = None
        self.origin_wall: float | None = None
        self.origin_sim: float | None = None
        self.stream = None
        self.log = (output / "trajectory.jsonl").open("w", encoding="utf-8")
        self.video = av.open(str(output / "video.mp4"), mode="w")

    def snapshot(self, model, data) -> None:
        """Keep the integration state from just before the next control step."""
        self.before_step = {
            "schema": STATE_SCHEMA, "mujoco_version": mujoco.__version__,
            "model_sha256": self.model_sha256, "values": integration_state(model, data).tolist(),
        }

    def commanded(self, left, right) -> None:
        self.targets = {"left": np.asarray(left, dtype=float).tolist(), "right": np.asarray(right, dtype=float).tolist()}

    def capture(self, env, frames: dict) -> None:
        """Add one camera sample and telemetry row per 50 ms simulator step."""
        now = time.monotonic()
        sim = float(env.data.time)
        if self.origin_wall is None:
            self.origin_wall, self.origin_sim = now, sim
        elif self.rows and abs((sim - self.origin_sim) - self.rows[-1]["sim_seconds"]) <= 1e-6:
            return  # A viewer refresh while inference holds simulation time.
        elif abs((sim - self.origin_sim) - len(self.rows) / FPS) > 1e-6:
            raise RuntimeError("Expected one camera sample per 50 ms simulator step.")
        # Pack one sample from each camera into the same video frame.
        mosaic = np.concatenate([frames[camera] for camera in CAMERAS], axis=1)
        frame = av.VideoFrame.from_ndarray(mosaic, format="rgb24")
        if self.stream is None:
            self.stream = self.video.add_stream("libx264", rate=FPS)
            self.stream.width, self.stream.height = frame.width, frame.height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "ultrafast", "crf": "23"}
        model, data = env.model, env.data
        row = {
            "frame": len(self.rows),
            "sim_seconds": len(self.rows) / FPS,
            "wall_seconds": round(now - self.origin_wall, 6),
            "measured_joint_rad_and_gripper": {side: values.tolist() for side, values in env.state().items()},
            "commanded_joint_rad_and_gripper": self.targets,
            "physics_before_step": self.before_step,
            # One "<object>_contacts" list per movable object in the scene, e.g. "apple_contacts".
            **{contacts_key(name): body_contacts(model, data, name) for name in self.scene.objects},
            "contact_samples": sample_contacts(model, data),
            "object_xyz_m": {name: data.body(name).xpos.tolist()
                             for name in (*self.scene.objects, *self.scene.supports)},
        }
        line = json.dumps(row, allow_nan=False)
        for packet in self.stream.encode(frame):
            self.video.mux(packet)
        self.log.write(line + "\n")
        self.log.flush()
        self.rows.append(row)
        if self.watchdog is not None and not self.home_stop:
            self.watchdog.observe(row)
            if self.watchdog.should_stop():
                self.home_stop = True
                print(f"  Arms back near the initial pose for more than {self.watchdog.home_seconds:g} s "
                      f"(since {self.watchdog.near_since:.2f} s sim). Stopping the trial.", flush=True)
                if self.on_home is not None:
                    self.on_home()

    def close(self, metadata: dict) -> Path:
        """Flush the video, then write run.json and replay.html. Returns the recording folder."""
        try:
            if self.stream is not None:
                for packet in self.stream.encode():
                    self.video.mux(packet)
        finally:
            self.video.close()
            self.log.close()
        metadata = {**metadata, "frames": len(self.rows), "fps": FPS, "includes_warmup": True,
                    "video_camera_order": list(CAMERAS), "video_layout": "horizontal",
                    "scene_objects": list(self.scene.objects), "scene_supports": list(self.scene.supports)}
        if self.watchdog is not None:
            metadata["home_rule"] = self.watchdog.summary()
        (self.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if self.rows:
            write_replay(self.output / "replay.html", self.rows)
        return self.output


class ViewportPublisher:
    """Save the free 3D viewer angle as a JPEG for the Web page, at most every 0.16 s."""

    def __init__(self, path: Path, lookat, distance: float, azimuth: float, elevation: float) -> None:
        from PIL import Image

        self.image = Image
        self.path = path
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = lookat
        self.camera.distance, self.camera.azimuth, self.camera.elevation = distance, azimuth, elevation
        self.last = float("-inf")

    def publish(self, env, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < VIEWPORT_PERIOD_S:
            return
        if env.renderer is None:
            env.renderer = mujoco.Renderer(env.model, height=480, width=640)
        env.renderer.update_scene(env.data, camera=self.camera)
        pending = self.path.with_suffix(".tmp")
        self.image.fromarray(env.renderer.render()).save(pending, format="JPEG", quality=80)
        pending.replace(self.path)
        self.last = now


def recording_env(base: type) -> type:
    """Subclass the upstream YamEnv with a scene layout, an optional recorder and viewport hook."""

    class RecordingEnv(base):
        scene: Scene = SCENES[DEFAULT_SCENE_ID]
        recorder: TrialRecorder | None = None
        viewport: ViewportPublisher | None = None

        def reset(self):
            state = super().reset()
            if any(self.scene.apple_offset_xyz_m):
                self.data.joint("apple_free").qpos[:3] += self.scene.apple_offset_xyz_m
                mujoco.mj_forward(self.model, self.data)
            return state

        def step(self, left, right):
            recorder = self.recorder
            if recorder is not None:
                recorder.snapshot(self.model, self.data)
            state = super().step(left, right)
            if recorder is not None:
                recorder.commanded(left, right)
            return state

        def images(self):
            frames = super().images()
            if self.viewport is not None:
                self.viewport.publish(self)
            if self.recorder is not None:
                self.recorder.capture(self, frames)
            return frames

        def hold(self) -> None:
            """Hold measured positions, not the last unfinished policy target (as viewer_workspace.py does)."""
            for i in range(self.model.nu):
                joint = self.model.actuator_trnid[i, 0]
                self.data.ctrl[i] = np.clip(self.data.qpos[self.model.jnt_qposadr[joint]],
                                            *self.model.actuator_ctrlrange[i])
            self.data.qvel[:] = 0

    return RecordingEnv


def load_upstream():
    """Import the Dream YAM Sim example modules, which import their siblings by bare name."""
    import importlib.util
    import sys

    spec = importlib.util.find_spec("dream_examples.yam_mujoco")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Dream YAM Sim is missing. Run with ~/dream-env/bin/python.")
    location = str(next(iter(spec.submodule_search_locations)))
    if location not in sys.path:
        sys.path.insert(0, location)
    import dream_control
    import yam_env

    return dream_control, yam_env
