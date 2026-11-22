"""The live viewport camera: clamping, and the publisher applying a view set from another thread. No API calls."""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from dreamlabs.viewport_camera import clean_camera

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None and importlib.util.find_spec("av") is not None


class CleanCameraTests(unittest.TestCase):
    def test_wraps_and_clamps(self) -> None:
        view = clean_camera({"azimuth": 725, "elevation": 120, "distance": 0.01, "lookat": [9, -9, 1]})
        self.assertEqual(view, {"azimuth": 5.0, "elevation": 89.0, "distance": 0.15, "lookat": [5.0, -5.0, 1.0]})

    def test_rejects_bad_values(self) -> None:
        good = {"azimuth": 0, "elevation": 0, "distance": 1, "lookat": [0, 0, 0]}
        for change in ({"azimuth": float("nan")}, {"distance": "1"}, {"lookat": [0, 0]}, {"lookat": None},
                       {"elevation": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                clean_camera({**good, **change})
        with self.assertRaises(ValueError):
            clean_camera([1, 2, 3])


@unittest.skipUnless(HAS_MUJOCO, "needs MuJoCo and PyAV")
class PublisherTests(unittest.TestCase):
    def test_new_view_is_applied_at_the_next_publish(self) -> None:
        import numpy as np

        from dreamlabs.trial_recorder import ViewportPublisher

        class Renderer:
            def __init__(self) -> None:
                self.cameras = []

            def update_scene(self, data, camera) -> None:
                self.cameras.append((camera.azimuth, camera.elevation, camera.distance, list(camera.lookat)))

            def render(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        with TemporaryDirectory() as directory:
            publisher = ViewportPublisher(Path(directory) / "v.jpg", [0, 0, 0.8], 1.5, 90, -30)
            env = SimpleNamespace(renderer=Renderer(), data=None)
            publisher.publish(env)
            publisher.set_camera({"azimuth": 10, "elevation": -45, "distance": 0.5, "lookat": [0.1, 0, 0.7]})
            publisher.publish(env)  # within the 0.16 s period, but a new view is shown at once
            publisher.set_camera(None)
            publisher.publish(env)
            self.assertTrue((Path(directory) / "v.jpg").is_file())
        first, moved, reset = env.renderer.cameras
        self.assertEqual(moved, (10.0, -45.0, 0.5, [0.1, 0.0, 0.7]))
        self.assertEqual(first, reset)


if __name__ == "__main__":
    unittest.main()
