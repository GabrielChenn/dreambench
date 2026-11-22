"""sim_workspace.Workspace in the real MuJoCo scene, with Dream-1 replaced by a scripted policy. No API calls.

The fake dream_control.main only uses what the real one does: env.state/step/images and stop_event.is_set().
"""

import argparse
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

HAS_SIM = importlib.util.find_spec("dream_examples") is not None and importlib.util.find_spec("mujoco") is not None
CONFIG = {"schema_version": 1, "model": "beta", "seconds": 60, "auto_stop_home_seconds": 0.5,
          "comparison_variable": "action_limit",
          "baseline": {"prompt": "put orange into the bowl", "action_limit": 48},
          "candidate": {"prompt": "put orange into the bowl", "action_limit": 24}}


class ScriptedPolicy:
    """Reach out with both shoulders and elbows, come back home, then hold until stopped."""

    def __init__(self, on_step=None, max_steps: int = 400) -> None:
        self.on_step = on_step
        self.max_steps = max_steps
        self.calls = 0

    def main(self, options, shared_env=None, shared_viewer=None, stop_event=None, collision_view=None):
        import numpy as np

        self.calls += 1
        env = shared_env
        for step in range(self.max_steps):
            if stop_event is not None and stop_event.is_set():
                return
            out = 20 <= step < 60  # 2 s out, then home
            target = np.r_[0.0, 0.9 if out else 0.0, 0.9 if out else 0.0, 0.0, 0.0, 0.0, 1.0]
            env.step(target, target)
            env.images()
            if self.on_step:
                self.on_step(step)
        print("  Dream · 0.10s RTT · 3 requests · 400 actions", flush=True)


@unittest.skipUnless(HAS_SIM, "Dream YAM simulator not installed")
class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sim_workspace
        from dreamlabs import trial_recorder

        cls.sim_workspace = sim_workspace
        cls.trial_recorder = trial_recorder
        cls.dream_control, cls.yam_env = trial_recorder.load_upstream()

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        args = argparse.Namespace(output_dir=self.output, viewport_jpeg=self.output / "viewport.jpg", dry_run=False)
        self.policy = ScriptedPolicy()
        self.workspace = self.sim_workspace.Workspace(args, self.policy, self.yam_env, self.trial_recorder)
        self.addCleanup(self.workspace.close)

    def run_experiment(self, stop: threading.Event, config: dict) -> dict:
        with redirect_stdout(io.StringIO()) as log:
            self.workspace.experiment({"job_id": "j", "config": json.loads(json.dumps(config)), "stop_event": stop})
        events = [json.loads(line[len("@@sim "):]) for line in log.getvalue().splitlines() if line.startswith("@@sim ")]
        finished = events[-1]
        self.assertEqual(finished["event"], "experiment_finished", log.getvalue())
        return json.loads(Path(finished["report"]).read_text())

    def test_home_auto_stop_is_failed(self) -> None:
        report = self.run_experiment(threading.Event(), CONFIG)
        for label in ("baseline", "candidate"):
            trial = report[label]
            self.assertEqual((trial["stop_reason"], trial["verdict"], trial["outcome"]),
                             ("home", "failed", "task-failure"))
            self.assertEqual(trial["reported_api_requests"], None)  # stopped before the scripted summary line
            self.assertTrue(trial["home_rule"]["stop"])
            self.assertEqual(trial["diagnostics"]["stage"], "untouched")
            self.assertTrue(Path(trial["replay"]).is_file())
        self.assertEqual(report["baseline"]["initial_state"], report["candidate"]["initial_state"])
        self.assertTrue((self.output / "viewport.jpg").is_file())

    def test_stop_saves_the_trial_and_skips_the_rest(self) -> None:
        stop = threading.Event()
        self.policy.on_step = lambda step: stop.set() if step == 30 else None
        report = self.run_experiment(stop, {**CONFIG, "auto_stop_home_seconds": 0})
        baseline, candidate = report["baseline"], report["candidate"]
        self.assertEqual((baseline["stop_reason"], baseline["verdict"]), ("operator", "failed"))
        self.assertTrue(Path(baseline["replay"]).is_file())
        self.assertEqual((candidate["outcome"], candidate["verdict"]), ("not-run", None))
        self.assertTrue(report["stopped"])
        self.assertEqual(report["decision"], "insufficient-evidence")
        self.assertEqual(self.policy.calls, 1)

    def test_time_limit_without_home_stop_fails_and_counts_requests(self) -> None:
        self.policy.max_steps = 80
        report = self.run_experiment(threading.Event(), {**CONFIG, "auto_stop_home_seconds": 0})
        self.assertEqual((report["baseline"]["stop_reason"], report["baseline"]["verdict"]), ("finished", "failed"))
        self.assertEqual(report["baseline"]["reported_api_requests"], 3)

    def test_reset_robot_and_reset(self) -> None:
        import numpy as np

        env = self.workspace.env
        env.reset()
        initial = env.data.body("orange").xpos.copy()
        bent = np.r_[0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 1.0]
        for _ in range(40):
            env.step(bent, bent)
        env.data.joint("orange_free").qpos[0] += 0.05
        __import__("mujoco").mj_forward(env.model, env.data)  # update xpos after editing qpos
        orange = env.data.body("orange").xpos.copy()
        with redirect_stdout(io.StringIO()):
            self.workspace.reset_robot()
        state = env.state()
        self.assertLess(max(abs(state[side][1]) for side in ("left", "right")), 0.1)
        # The objects stay where they were moved; only the arms go home.
        self.assertGreater(abs(env.data.body("orange").xpos[0] - initial[0]), 0.03)
        self.assertLess(abs(env.data.body("orange").xpos[0] - orange[0]), 0.02)
        with redirect_stdout(io.StringIO()):
            self.workspace.reset()
        self.assertTrue(np.allclose(env.data.body("orange").xpos, initial, atol=1e-6))

    def test_any_event(self) -> None:
        first, second = threading.Event(), threading.Event()
        either = self.sim_workspace.AnyEvent(first, second)
        self.assertFalse(either.is_set())
        second.set()
        self.assertTrue(either.is_set())


class CommandsTests(unittest.TestCase):
    """The stdin reader: stop-like commands set every pending experiment's token at once."""

    def test_stop_and_reset_set_pending_tokens(self) -> None:
        import sim_workspace

        lines = [json.dumps({"command": "experiment", "job_id": "a"}), "not json",
                 json.dumps({"command": "stop"}), json.dumps({"command": "reset"})]
        commands = sim_workspace.Commands(io.StringIO("\n".join(lines) + "\n"))
        commands.read()
        experiment = commands.next(0)
        self.assertTrue(experiment["stop_event"].is_set())
        self.assertEqual([commands.next(0)["command"], commands.next(0)["command"]], ["reset", "quit"])
        self.assertIsNone(commands.next(0))

    def test_camera_is_handled_on_the_reader_thread(self) -> None:
        import sim_workspace

        commands = sim_workspace.Commands(io.StringIO(json.dumps({"command": "camera", "reset": True}) + "\n"))
        seen = []
        commands.on_camera = seen.append
        commands.read()
        self.assertEqual(seen, [{"command": "camera", "reset": True}])
        self.assertEqual(commands.next(0)["command"], "quit")  # nothing queued for the busy main loop


if __name__ == "__main__":
    unittest.main()
