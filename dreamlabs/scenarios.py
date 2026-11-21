"""Registered YAM tabletop layouts. Every layout holds the same objects and supports."""

from dataclasses import dataclass


DEFAULT_SCENE_ID = "yam-tabletop-default"


@dataclass(frozen=True)
class Scene:
    apple_offset_xyz_m: tuple[float, float, float]


SCENES = {
    DEFAULT_SCENE_ID: Scene((0.0, 0.0, 0.0)),
    "yam-tabletop-offset": Scene((-0.04, 0.025, 0.0)),
}
