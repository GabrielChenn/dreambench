"""Registered YAM tabletop layouts and their supported task goals."""

from dataclasses import dataclass


DEFAULT_SCENE_ID = "yam-tabletop-default"


@dataclass(frozen=True)
class Scene:
    apple_offset_xyz_m: tuple[float, float, float]
    task_ids: tuple[str, ...]


SCENES = {
    DEFAULT_SCENE_ID: Scene((0.0, 0.0, 0.0), ("apple-in-bowl",)),
    "yam-tabletop-offset": Scene((-0.04, 0.025, 0.0), ("apple-in-bowl",)),
}
