"""Replay page, collision categories, and the 3D review restored from pre-step snapshots. No API calls."""

import http.client
import importlib.util
import io
import json
import sys
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from dreamlabs.contact_events import COLLISIONS, contact_category
from dreamlabs.replay_page import CATEGORY_COLORS, replay_html
from dreamlabs.web_jobs import SimWorkspace
from dreamlabs.yolo_status import yolo_note, yolo_unavailable
from web_dashboard import ThreadingHTTPServer, live_replay, make_handler, recording_dir, review_camera

HAS_SIM = importlib.util.find_spec("dream_examples") is not None and importlib.util.find_spec("mujoco") is not None


class ContactCategoryTests(unittest.TestCase):
    def test_categories(self) -> None:
        cases = {
            ("g", "left_link3", "g", "left_link5"): "arm_self",
            ("g", "left_link3", "g", "right_link5"): "arm_arm",
            ("table", "world", "g", "left_link6"): "arm_table",
            ("g", "orange", "g", "left_tip_left"): "grasp_contact",
            ("g", "water_bottle", "g", "right_tip_right"): "grasp_contact",
            ("g", "apple", "g", "bowl"): "apple_bowl",
            ("g", "right_link6", "g", "bowl"): "arm_object",
            ("g", "left_tip_left", "g", "bowl"): "arm_object",  # a fingertip pushing a fixed container
            ("g", "banana", "g", "plate"): "object_support",
            ("table", "world", "g", "orange"): None,  # resting on the table is not tracked
            ("", "world", "g", "left_link1"): None,  # the floor
        }
        for (geom_a, body_a, geom_b, body_b), expected in cases.items():
            with self.subTest(a=body_a, b=body_b):
                self.assertEqual(contact_category(geom_a, body_a, geom_b, body_b), expected)
                self.assertEqual(contact_category(geom_b, body_b, geom_a, body_a), expected)

    def test_every_lane_has_a_color(self) -> None:
        self.assertTrue(set(COLLISIONS) <= set(CATEGORY_COLORS))


class ReplayPageTests(unittest.TestCase):
    def test_controls_and_escaping(self) -> None:
        page = replay_html([{"frame": 0, "sim_seconds": 0.0, "note": "__ROWS__ </script><b>"}])
        for control in ('id="timeline"', 'id="events"', 'id="review-image"', 'id="next-collision"',
                        'id="previous-collision"', 'id="follow"', 'id="contact-values"', 'id="object-values"',
                        'id="play"', 'id="speed"', 'id="video"'):
            self.assertIn(control, page)
        self.assertNotIn("</script><b>", page)
        self.assertEqual(page.count("__ROWS__"), 1)  # the recorded text, not a second substitution


class ReviewRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.recording = self.root / "comparison-a" / "baseline" / "replay-x"
        self.recording.mkdir(parents=True)
        (self.recording / "trajectory.jsonl").write_text(json.dumps({"frame": 0, "sim_seconds": 0.0}) + "\n")
        (self.recording / "replay.html").write_text("old page")

    def test_recording_dir_stays_under_outputs(self) -> None:
        self.assertEqual(recording_dir(self.root, "comparison-a/baseline/replay-x"), self.recording)
        for bad in ("../x", "/etc", "comparison-a", "comparison-a/baseline", "", None, "a\x00b"):
            with self.subTest(bad=bad):
                self.assertIsNone(recording_dir(self.root, bad))

    def test_camera_query(self) -> None:
        camera = review_camera({"az": ["45"], "el": ["-20"], "dist": ["nan"], "lx": ["0"], "ly": ["0.1"], "lz": ["0.8"]})
        self.assertEqual(camera, {"azimuth": 45.0, "elevation": -20.0, "lookat": [0.0, 0.1, 0.8]})
        self.assertEqual(review_camera({"az": ["x"], "lx": ["1"]}), {})

    def test_live_replay_rebuilds_old_pages(self) -> None:
        page = live_replay(self.recording / "replay.html")
        self.assertIn(b'id="review-image"', page)
        self.assertIsNone(live_replay(self.recording / "video.mp4"))
        self.assertIsNone(live_replay(self.recording / "replay-yolo.html"))  # no detections file
        self.assertIn(b"YOLO was not run", page)

    def test_live_replay_shows_yolo_boxes_and_failures(self) -> None:
        record = {"frame": 0, "sim_seconds": 0.0, "ran": True, "camera_size_px": [640, 480],
                  "cameras": {"top": [{"label": "apple", "bbox_xyxy_px": [1, 2, 3, 4], "confidence": 0.9}],
                              "left": [], "right": []}}
        (self.recording / "run.json").write_text(json.dumps({"yolo": {"status": "failed", "error": "no <weights>"}}))
        page = live_replay(self.recording / "replay.html").decode()
        self.assertIn("requested for this run but failed: no &lt;weights&gt;", page)
        (self.recording / "detections-yolo.jsonl").write_text(json.dumps(record) + "\n")
        for name in ("replay.html", "replay-yolo.html"):  # both pages draw the boxes on the camera views
            page = live_replay(self.recording / name).decode()
            self.assertIn('"bbox_xyxy_px": [1, 2, 3, 4]', page)
            self.assertIn("detection ran on 1 of 1 frames", page)

    def test_yolo_unavailable_names_what_is_missing(self) -> None:
        reason = yolo_unavailable(self.root / "missing.pt")
        self.assertIn("missing.pt", reason)
        self.assertEqual(yolo_note(self.recording, None), "No YOLO detections for this recording (YOLO was not run).")

    def test_review_endpoint_without_renderer(self) -> None:
        sim = SimWorkspace(self.root, Path(sys.executable), Path(__file__))
        self.addCleanup(sim.close)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.root, sim))
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        port = server.server_port

        def get(path: str) -> int:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            self.addCleanup(connection.close)
            connection.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
            response = connection.getresponse()
            response.read()
            return response.status

        self.assertEqual(get("/api/review/summary?recording=comparison-a/baseline/replay-x"), 503)
        self.assertEqual(get("/api/review/summary?recording=../../etc"), 404)
        self.assertEqual(get("/artifacts/comparison-a/baseline/replay-x/replay.html"), 200)


@unittest.skipUnless(HAS_SIM, "Dream YAM simulator not installed")
class ReviewSceneTests(unittest.TestCase):
    """Record a short trial in the real scene with a scripted policy, then restore and render it."""

    @classmethod
    def setUpClass(cls) -> None:
        import argparse

        import sim_workspace
        from dreamlabs import trial_recorder
        from dreamlabs.review_scene import ReviewScene
        from test_sim_workspace import CONFIG, ScriptedPolicy

        cls.directory = TemporaryDirectory()
        output = Path(cls.directory.name)
        _, yam_env = trial_recorder.load_upstream()
        policy = ScriptedPolicy(max_steps=90)
        args = argparse.Namespace(output_dir=output, viewport_jpeg=None, dry_run=False)
        workspace = sim_workspace.Workspace(args, policy, yam_env, trial_recorder)
        stop = threading.Event()
        with redirect_stdout(io.StringIO()):
            workspace.experiment({"job_id": "j", "stop_event": stop,
                                  "config": {**json.loads(json.dumps(CONFIG)), "auto_stop_home_seconds": 0}})
        workspace.close()
        cls.recording = next(output.glob("comparison-*/baseline/replay-*"))
        cls.camera = {"lookat": [float(v) for v in yam_env.VIEWER_LOOKAT], "distance": 1.2,
                      "azimuth": 90.0, "elevation": -25.0}
        cls.scene = ReviewScene(yam_env.build_xml, cls.camera)
        cls.yam_env = yam_env

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_restored_state_matches_the_recording(self) -> None:
        rows = [json.loads(line) for line in (self.recording / "trajectory.jsonl").read_text().splitlines()]
        recording = self.scene.load(self.recording)
        # Restoring frame n+1's pre-step state gives the state recorded after step n.
        for frame in (10, 40, 70):
            self.scene.restore(recording, frame + 1)
            measured = rows[frame]["measured_joint_rad_and_gripper"]
            for side in ("left", "right"):
                joints = [self.scene.data.joint(f"{side}_joint{i}").qpos[0] for i in range(1, 7)]
                for restored, recorded in zip(joints, measured[side][:6]):
                    self.assertAlmostEqual(restored, recorded, places=9)

    def test_summary_and_render(self) -> None:
        summary = self.scene.summary(self.recording)
        self.assertEqual(summary["restorable"], summary["frames"])
        self.assertEqual(set(summary["tips"][0]), {"left", "right"})
        self.assertTrue(all(isinstance(frame, list) for frame in summary["contacts"]))
        # The scripted arms touch nothing, so only resting contacts (objects on the table) remain: untracked.
        self.scene.restore(self.scene.load(self.recording), 50)
        resting = [item for item in self.scene.contacts() if item["category"] is None]
        self.assertTrue(any("orange" in item["bodies"] and "world" in item["bodies"] for item in resting))
        image = self.scene.render(self.recording, 50, self.camera)
        self.assertTrue(image.startswith(b"\xff\xd8") and len(image) > 10_000)

    def test_out_of_range_and_foreign_model_are_refused(self) -> None:
        from dreamlabs.review_scene import ReviewError

        with self.assertRaisesRegex(ReviewError, "outside"):
            self.scene.render(self.recording, 10_000, self.camera)
        with TemporaryDirectory() as directory:
            copy = Path(directory) / "replay-copy"
            copy.mkdir()
            rows = [json.loads(line) for line in (self.recording / "trajectory.jsonl").read_text().splitlines()]
            rows[0]["physics_before_step"]["model_sha256"] = "0" * 64
            (copy / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ReviewError, "different YAM scene model"):
                self.scene.load(copy)
            rows = [{key: value for key, value in row.items() if key != "physics_before_step"} for row in rows]
            (copy / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ReviewError, "predates physics snapshots"):
                self.scene.load(copy)

    def test_worker_process_round_trip(self) -> None:
        from dreamlabs.review_client import ReviewRenderer

        root = Path(__file__).resolve().parents[1]
        renderer = ReviewRenderer(Path(sys.executable), root / "review_worker.py")
        self.addCleanup(renderer.close)
        self.assertEqual(renderer.summary(self.recording)["frames"], self.scene.summary(self.recording)["frames"])
        image = renderer.render(self.recording, 5, {"azimuth": 45.0, "distance": 0.8}, contacts=True, other=True,
                                trails=True)
        self.assertTrue(image.startswith(b"\xff\xd8"))
        with self.assertRaisesRegex(ValueError, "outside"):
            renderer.render(self.recording, 99_999, {}, contacts=True, other=False, trails=True)


if __name__ == "__main__":
    unittest.main()
