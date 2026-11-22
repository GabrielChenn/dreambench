"""Random scene layouts (every movable object moved) and the generalization table. No API calls."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import compare_replays
from compare_replays import conditions, validate_config
from dreamlabs.perturbation import ObjectOffset, generalization, sample_layouts, validate_random_starts
from dreamlabs.scoring import OBJECTS
import web_dashboard

HAS_SIM = importlib.util.find_spec("dream_examples") is not None and importlib.util.find_spec("mujoco") is not None
SINGLE = {"schema_version": 1, "model": "beta", "seconds": 60,
          "baseline": {"prompt": "put apple into the bowl", "action_limit": 48}}
PAIR = {**SINGLE, "comparison_variable": "action_limit",
        "candidate": {"prompt": "put apple into the bowl", "action_limit": 24}}
STARTS = {"count": 5, "max_offset_m": 0.04, "seed": 7}
SEED_WITH_OVERLAP = 5  # at ±7 cm, 6 layouts: this seed draws objects that overlap and are redrawn


def copy(config: dict) -> dict:
    return json.loads(json.dumps(config))


def offset_of(offsets: list[dict], name: str) -> dict:
    return next(item for item in offsets if item["object"] == name)


class RandomLayoutTests(unittest.TestCase):
    def test_validation(self) -> None:
        self.assertEqual(validate_random_starts({"count": 3, "seed": 1}), {"count": 3, "max_offset_m": 0.04, "seed": 1})
        for raw in ([], {"count": 0, "seed": 1}, {"count": 10, "seed": 1}, {"count": 3}, {"count": 3, "seed": -1},
                    {"count": 3, "seed": 1.5}, {"count": True, "seed": 1}, {"count": 3, "seed": 1, "max_offset_m": 0.2},
                    {"count": 3, "seed": 1, "max_offset_m": float("nan")}, {"count": 3, "seed": 1, "extra": 1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_random_starts(raw)

    def test_config_keeps_random_starts_and_stays_idempotent(self) -> None:
        loaded = validate_config({**copy(SINGLE), "random_starts": STARTS})
        self.assertEqual(loaded["random_starts"], STARTS)
        self.assertEqual(validate_config(copy(loaded)), loaded)
        self.assertNotIn("random_starts", validate_config({**copy(SINGLE), "random_starts": None}))

    def test_each_layout_moves_every_object_seeded_and_bounded(self) -> None:
        first, _ = sample_layouts(STARTS, OBJECTS)
        again, _ = sample_layouts(STARTS, OBJECTS)
        other, _ = sample_layouts({**STARTS, "seed": 8}, OBJECTS)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(len(set(first)), 5)
        for layout in first:
            self.assertEqual(tuple(offset.object for offset in layout), OBJECTS)
            self.assertTrue(all(abs(o.dx_m) <= 0.04 and abs(o.dy_m) <= 0.04 for o in layout))
        # objects are moved independently, not by one shared offset
        self.assertTrue(any(len({(o.dx_m, o.dy_m) for o in layout}) > 1 for layout in first))

    def test_sampling_redraws_conflicts_and_gives_up(self) -> None:
        def banana_left(layout):
            return "inside orange" if layout[1].dx_m < 0 else None

        layouts, rejected = sample_layouts(STARTS, OBJECTS, banana_left)
        self.assertEqual(len(layouts), 5)
        self.assertTrue(all(layout[1].dx_m >= 0 for layout in layouts))
        self.assertTrue(rejected and all(offset_of(item["offsets"], "banana")["dx_m"] < 0 for item in rejected))
        with self.assertRaisesRegex(ValueError, "usable layouts"):
            sample_layouts(STARTS, OBJECTS, lambda layout: "inside bowl")

    def test_conditions_cross_scenes_with_layouts(self) -> None:
        config = validate_config({**copy(PAIR), "scene_ids": ["yam-tabletop-default", "yam-tabletop-offset"],
                                  "random_starts": {**STARTS, "count": 2}})
        planned, sampling = conditions(config)
        self.assertEqual([folder for folder, _ in planned], [
            "yam-tabletop-default/start-1", "yam-tabletop-default/start-2",
            "yam-tabletop-offset/start-1", "yam-tabletop-offset/start-2"])
        self.assertEqual(planned[0][1]["object_offsets"], sampling["yam-tabletop-default"]["accepted"][0])
        self.assertEqual({item["object"] for item in planned[0][1]["object_offsets"]}, set(OBJECTS))
        self.assertEqual(len(conditions(validate_config(copy(PAIR)))[0]), 1)

    def test_generalization_counts_each_verdict(self) -> None:
        pairs = [{"baseline": {"outcome": "success", "verdict": "successed"}},
                 {"baseline": {"outcome": "task-failure", "verdict": "failed"}},
                 {"baseline": {"outcome": "task-failure", "verdict": "failed"}},
                 {"baseline": {"outcome": "invalid-start", "verdict": None}}]
        self.assertEqual(generalization(pairs)["totals"], {"baseline": {
            "passed": 1, "failed": 2, "scored": 3, "conditions": 4}})


class RandomLayoutExperimentTests(unittest.TestCase):
    """run_experiment with a fake trial runner and preflight, then the results page."""

    def run_layouts(self, config: dict, verdict_for, conflict_for=lambda offsets: None):
        calls = []

        def trial(folder: Path, trial_config: dict, label: str) -> Path:
            calls.append((trial_config["object_offsets"], label))
            folder = folder / label / "replay-test"
            folder.mkdir(parents=True)
            (folder / "replay.html").write_text("<html></html>")
            return folder

        def assess(folder: Path) -> dict:
            offsets, label = calls[-1]
            verdict = verdict_for(offsets, label)
            return {"outcome": "success" if verdict == "successed" else "task-failure", "verdict": verdict,
                    "reason": verdict, "evidence_sim_s": None, "replay": str(folder / "replay.html"),
                    "actions": 48 if label == "baseline" else 24, "prompt": "put apple into the bowl",
                    "task_id": "apple-in-bowl", "scene_id": "yam-tabletop-default", "scorer": "s", "model": "beta",
                    "seconds": 60, "initial_state": {"offsets": offsets}, "object_offsets": offsets}

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        with patch.object(compare_replays, "assess", side_effect=assess), redirect_stdout(io.StringIO()):
            path = compare_replays.run_experiment(validate_config(copy(config)), root / "comparison-test", trial,
                                                  preflight=lambda c: conflict_for(c["object_offsets"]))
        return root, path, json.loads(path.read_text()), calls

    def test_single_run_table_counts_every_verdict_with_replays_last(self) -> None:
        def verdict_for(offsets, label):
            dx = offset_of(offsets, "apple")["dx_m"]
            return "successed" if dx <= 0 else "failed"

        root, path, report, calls = self.run_layouts(
            {**SINGLE, "random_starts": STARTS}, verdict_for,
            conflict_for=lambda offsets: "the banana starts inside orange" if offset_of(offsets, "banana")["dy_m"] < -0.02 else None)
        self.assertEqual(len(calls), 5)  # rejected draws are replaced, never run
        self.assertTrue(all(offset_of(offsets, "banana")["dy_m"] >= -0.02 for offsets, _ in calls))
        info = report["generalization"]
        self.assertEqual(info["random_starts"], STARTS)
        self.assertTrue(info["sampling"]["yam-tabletop-default"]["rejected"])
        verdicts = [verdict_for(offsets, label) for offsets, label in calls]
        totals = info["totals"]["baseline"]
        self.assertEqual((totals["passed"], totals["failed"]), (verdicts.count("successed"), verdicts.count("failed")))
        self.assertTrue((path.parent / "start-5").is_dir())
        page = web_dashboard.dashboard(root, None)
        panel = page[page.index("Generalization: random object layouts"):page.index("</table>")]
        self.assertIn("5 random layouts, each object moved uniformly within ±4.00 cm, seed 7", panel)
        for name in ("apple (judged)", "banana", "orange", "water bottle"):
            self.assertIn(name, panel)
        self.assertEqual(panel.count("Open run replay"), 5)
        for row in panel.split("<tr>")[2:7]:  # body rows: the replay link is the last cell
            self.assertTrue(row.split("</tr>")[0].endswith("Open run replay</a></td>"), row)
        self.assertIn(f"{totals['failed']} failed", panel)
        self.assertNotIn("timeout stopped", page)
        self.assertNotIn("<circle", page)

    def test_same_seed_same_layouts(self) -> None:
        _, _, _, first = self.run_layouts({**SINGLE, "random_starts": STARTS}, lambda *a: "successed")
        _, _, _, again = self.run_layouts({**SINGLE, "random_starts": STARTS}, lambda *a: "successed")
        self.assertEqual(first, again)

    def test_paired_layouts_report_both_arms(self) -> None:
        root, _, report, calls = self.run_layouts(
            {**PAIR, "random_starts": {**STARTS, "count": 2}},
            verdict_for=lambda offsets, label: "successed" if label == "candidate" else "failed")
        self.assertEqual(len(calls), 4)
        self.assertEqual(report["decision"], "retest-candidate")
        totals = report["generalization"]["totals"]
        self.assertEqual((totals["baseline"]["passed"], totals["candidate"]["passed"]), (0, 2))
        page = web_dashboard.dashboard(root, None)
        self.assertIn("<th>Baseline</th><th>Candidate</th><th>Replay</th>", page)
        panel = page[page.index("Generalization: random object layouts"):page.index("</table>")]
        self.assertEqual(panel.count("Open baseline replay"), 2)
        self.assertEqual(panel.count("Open candidate replay"), 2)


@unittest.skipUnless(HAS_SIM, "Dream YAM simulator not installed")
class RandomLayoutSimulationTests(unittest.TestCase):
    """The real MuJoCo scene, with Dream-1 replaced by the scripted policy."""

    @classmethod
    def setUpClass(cls) -> None:
        from dreamlabs import trial_recorder

        cls.trial_recorder = trial_recorder
        cls.dream_control, cls.yam_env = trial_recorder.load_upstream()

    def env(self, layout=None):
        env = self.trial_recorder.recording_env(self.yam_env.YamEnv)()
        self.addCleanup(env.close)
        env.layout = layout
        env.reset()
        return env

    def test_layout_moves_each_object_and_is_repeatable(self) -> None:
        plain = self.env()
        layout = (ObjectOffset("apple", 0.04, -0.02), ObjectOffset("water_bottle", -0.03, 0.01))
        moved = self.env(layout)
        for offset in layout:
            delta = moved.data.body(offset.object).xpos - plain.data.body(offset.object).xpos
            self.assertAlmostEqual(delta[0], offset.dx_m, places=6)
            self.assertAlmostEqual(delta[1], offset.dy_m, places=6)
            self.assertAlmostEqual(delta[2], 0.0, places=6)
        self.assertEqual(moved.data.body("orange").xpos.tolist(), plain.data.body("orange").xpos.tolist())
        first = moved.data.qpos.tolist()
        moved.reset()
        self.assertEqual(moved.data.qpos.tolist(), first)
        self.assertIsNone(moved.start_conflict())

    def test_layout_conflicts(self) -> None:
        self.assertIn("bowl", self.env((ObjectOffset("apple", -0.10, -0.10),)).start_conflict())
        self.assertIn("orange", self.env((ObjectOffset("banana", 0.10, 0.0),)).start_conflict())
        with self.assertRaises(KeyError):
            self.env((ObjectOffset("no_such_body", 0.01, 0),))

    def test_workspace_runs_each_random_layout_and_redraws_overlaps(self) -> None:
        import argparse
        import sim_workspace
        from tests.test_sim_workspace import ScriptedPolicy

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name)
        policy = ScriptedPolicy(max_steps=10)
        workspace = sim_workspace.Workspace(argparse.Namespace(output_dir=output, viewport_jpeg=None, dry_run=False),
                                            policy, self.yam_env, self.trial_recorder)
        self.addCleanup(workspace.close)
        plain = {name: self.env().data.body(name).xpos.copy() for name in OBJECTS}
        spec = {"count": 6, "max_offset_m": 0.07, "seed": SEED_WITH_OVERLAP}
        config = {**copy(SINGLE), "auto_stop_home_seconds": 0, "random_starts": spec}
        with redirect_stdout(io.StringIO()) as log:
            workspace.experiment({"job_id": "j", "config": config, "stop_event": threading.Event()})
        finished = [json.loads(line[6:]) for line in log.getvalue().splitlines() if line.startswith("@@sim ")][-1]
        self.assertIsNone(finished["error"], log.getvalue())
        report = json.loads(Path(finished["report"]).read_text())
        self.assertEqual(policy.calls, 6)  # every requested layout ran; overlapping draws were replaced
        self.assertTrue(report["generalization"]["sampling"]["yam-tabletop-default"]["rejected"])
        for pair in report["scenes"]:
            trial = pair["baseline"]
            self.assertEqual(trial["object_offsets"], pair["object_offsets"])
            for item in pair["object_offsets"]:  # first sample is one control step after reset
                start = trial["initial_state"]["object_xyz_m"][item["object"]]
                self.assertAlmostEqual(start[0] - plain[item["object"]][0], item["dx_m"], delta=0.005)
                self.assertAlmostEqual(start[1] - plain[item["object"]][1], item["dy_m"], delta=0.005)
        self.assertIsNone(workspace.env.layout)


if __name__ == "__main__":
    unittest.main()
