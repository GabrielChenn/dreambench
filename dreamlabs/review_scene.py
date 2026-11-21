"""3D review of a recording: restore each frame's pre-step physics snapshot and render it in MuJoCo.

Every new trajectory row carries `physics_before_step`, the MuJoCo integration state saved just before
that frame's control step. Restoring it reproduces the exact scene (arms, fingers, every object's
position and orientation) at that instant, so the review can be rendered from any angle, with the
contacts MuJoCo finds in that state and the paths the grippers and objects took.

Import only in the Dream Labs environment (MuJoCo, NumPy, Pillow).
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from .contact_events import CATEGORIES, contact_category
from .frame_restore import STATE, STATE_SCHEMA, model_digest

WIDTH, HEIGHT = 960, 720
MAX_TRAIL_POINTS = 360
# RGBA per contact category; keep in step with CATEGORY_COLORS in replay_page.py.
COLORS = {
    "arm_table": (1.0, 0.25, 0.25, 1), "arm_arm": (0.95, 0.3, 0.95, 1), "arm_self": (0.7, 0.45, 1.0, 1),
    "arm_object": (1.0, 0.6, 0.1, 1), "grasp_contact": (0.3, 0.95, 0.45, 1), "apple_bowl": (0.35, 0.85, 1.0, 1),
    "object_support": (0.35, 0.85, 1.0, 1), None: (0.75, 0.75, 0.75, 1),
}
TRAIL_COLORS = {"left": (0.3, 0.65, 1.0), "right": (1.0, 0.4, 0.75), "apple": (1.0, 0.3, 0.3),
                "orange": (1.0, 0.6, 0.15), "banana": (1.0, 0.92, 0.3), "water_bottle": (0.4, 0.85, 1.0)}


@dataclass
class Recording:
    path: Path
    mtime: float
    states: list[np.ndarray | None]
    objects: list[dict]
    summary: dict | None = None


class ReviewError(ValueError):
    """A recording cannot be reviewed in 3D; the message is shown on the page."""


class ReviewScene:
    def __init__(self, build_xml, viewer_camera: dict) -> None:
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.digest = model_digest(self.model)  # before the render-only change below
        self.model.vis.global_.offwidth = max(WIDTH, int(self.model.vis.global_.offwidth))
        self.model.vis.global_.offheight = max(HEIGHT, int(self.model.vis.global_.offheight))
        self.data = mujoco.MjData(self.model)
        self.size = mujoco.mj_stateSize(self.model, STATE)
        self.renderer: mujoco.Renderer | None = None
        self.camera_defaults = viewer_camera
        self.cache: dict[Path, Recording] = {}
        names = [self.model.body(i).name for i in range(self.model.nbody)]
        self.tips = {side: [i for i, name in enumerate(names) if name.startswith(f"{side}_tip_")]
                     for side in ("left", "right")}

    # Loading ------------------------------------------------------------------------------------
    def load(self, path: Path) -> Recording:
        track = path / "trajectory.jsonl"
        if not track.is_file():
            raise ReviewError("This recording has no trajectory.jsonl.")
        mtime = track.stat().st_mtime
        cached = self.cache.get(path)
        if cached is not None and cached.mtime == mtime:
            return cached
        states, objects = [], []
        checked = False
        with track.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                snapshot = row.get("physics_before_step")
                state = None
                if isinstance(snapshot, dict) and snapshot.get("schema") == STATE_SCHEMA:
                    if not checked:
                        if snapshot.get("mujoco_version") != mujoco.__version__:
                            raise ReviewError(f"Recorded with MuJoCo {snapshot.get('mujoco_version')}; "
                                              f"this environment has {mujoco.__version__}.")
                        if snapshot.get("model_sha256") != self.digest:
                            raise ReviewError("Recorded with a different YAM scene model, so its snapshots "
                                              "cannot be restored here.")
                        checked = True
                    values = np.asarray(snapshot.get("values"), dtype=float)
                    if values.shape == (self.size,) and np.isfinite(values).all():
                        state = values
                states.append(state)
                objects.append(row.get("object_xyz_m") or {})
        if not any(state is not None for state in states):
            raise ReviewError("This recording predates physics snapshots, so it cannot be restored in 3D.")
        self.cache = {path: Recording(path, mtime, states, objects)}  # keep one recording in memory
        return self.cache[path]

    def restore(self, recording: Recording, frame: int) -> None:
        state = recording.states[frame]
        if state is None:
            raise ReviewError(f"Frame {frame} has no physics snapshot.")
        mujoco.mj_setState(self.model, self.data, state, STATE)
        mujoco.mj_forward(self.model, self.data)

    def contacts(self) -> list[dict]:
        """Contacts in the restored state, with the normal force MuJoCo solved for each (N)."""
        model, data = self.model, self.data
        force = np.zeros(6)
        result = []
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = [int(g) for g in contact.geom]
            bodies = [model.body(int(model.geom_bodyid[g])).name for g in geoms]
            category = contact_category(model.geom(geoms[0]).name, bodies[0], model.geom(geoms[1]).name, bodies[1])
            normal = 0.0
            if contact.efc_address >= 0:
                mujoco.mj_contactForce(model, data, index, force)
                normal = float(force[0])
            result.append({"category": category, "bodies": bodies, "normal_force_n": round(normal, 3),
                           "pos": contact.pos.tolist(), "normal": contact.frame[:3].tolist(),
                           "dist_mm": round(float(contact.dist) * 1000, 2)})
        return result

    def tip_positions(self) -> dict:
        return {side: np.mean([self.data.xpos[i] for i in ids], axis=0).round(4).tolist()
                for side, ids in self.tips.items() if ids}

    # Requests -----------------------------------------------------------------------------------
    def summary(self, path: Path) -> dict:
        """Per-frame gripper tips and categorized contacts, from the restored pre-step states."""
        recording = self.load(path)
        if recording.summary is None:
            tips, contacts = [], []
            for frame, state in enumerate(recording.states):
                if state is None:
                    tips.append(None)
                    contacts.append(None)
                    continue
                self.restore(recording, frame)
                tips.append(self.tip_positions())
                contacts.append([{key: item[key] for key in ("category", "bodies", "normal_force_n", "dist_mm")}
                                 for item in self.contacts() if item["category"] is not None])
            recording.summary = {
                "frames": len(recording.states), "restorable": sum(s is not None for s in recording.states),
                "tips": tips, "contacts": contacts, "categories": list(CATEGORIES),
                "camera": self.camera_defaults, "mujoco_version": mujoco.__version__,
            }
        return recording.summary

    def render(self, path: Path, frame: int, camera: dict, *, contacts: bool = True, other: bool = False,
               trails: bool = True) -> bytes:
        recording = self.load(path)
        if not 0 <= frame < len(recording.states):
            raise ReviewError(f"Frame {frame} is outside this recording.")
        self.restore(recording, frame)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH, max_geom=20000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = camera["lookat"]
        cam.distance, cam.azimuth, cam.elevation = camera["distance"], camera["azimuth"], camera["elevation"]
        self.renderer.update_scene(self.data, camera=cam)
        scene = self.renderer.scene
        if trails:
            self._trails(scene, recording, frame)
        if contacts:
            for item in self.contacts():
                if item["category"] is None and not other:
                    continue
                self._contact(scene, item)
        image = self.renderer.render()
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    # Scene decorations --------------------------------------------------------------------------
    @staticmethod
    def _add(scene, kind, size, pos, rgba) -> object | None:
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1
        return geom

    def _segment(self, scene, kind, width: float, start, end, rgba) -> None:
        geom = self._add(scene, kind, [0, 0, 0], [0, 0, 0], rgba)
        if geom is not None:
            mujoco.mjv_connector(geom, kind, width, np.asarray(start, dtype=float), np.asarray(end, dtype=float))

    def _contact(self, scene, item: dict) -> None:
        rgba = COLORS.get(item["category"], COLORS[None])
        radius = 0.004 if item["category"] is None else 0.007
        self._add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], item["pos"], rgba)
        force = item["normal_force_n"]
        if force > 0.05 and item["category"] is not None:
            length = min(0.12, 0.01 + 0.004 * force)  # 4 mm per newton, capped
            start = np.asarray(item["pos"])
            self._segment(scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.003, start,
                          start + np.asarray(item["normal"]) * length, rgba)

    def _trails(self, scene, recording: Recording, frame: int) -> None:
        summary = self.summary(recording.path)
        self.restore(recording, frame)  # summary() may have moved the state
        paths: dict[str, list] = {side: [tip[side] if tip else None for tip in summary["tips"]]
                                  for side in ("left", "right")}
        for name in TRAIL_COLORS:
            points = [objects.get(name) for objects in recording.objects]
            known = [p for p in points if p]
            if known and max(np.linalg.norm(np.subtract(p, known[0])) for p in known) > 0.01:
                paths[name] = points  # draw only objects that moved more than 10 mm
        for name, points in paths.items():
            rgb = TRAIL_COLORS[name]
            step = max(1, len(points) // MAX_TRAIL_POINTS)
            indices = list(range(0, len(points), step))
            for a, b in zip(indices, indices[1:]):
                if points[a] is None or points[b] is None:
                    continue
                past = b <= frame
                self._segment(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025 if past else 0.0012,
                              points[a], points[b], (*rgb, 0.95 if past else 0.25))
            if points[frame] is not None:
                self._add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.009, 0, 0], points[frame], (*rgb, 1))
