"""Judging ported from dreambench_bk: scorers, diagnostics, scene discovery, prompts, home rule. No API calls."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from compare_replays import load_config, validate_config
from dreamlabs import scene_inventory
from dreamlabs.home_watchdog import HomeWatchdog, joint_list
from dreamlabs.scene_inventory import FALLBACK, Inventory, discover, load_inventory
from dreamlabs.scoring import (CONTAINERS, FAILED, GENERIC_SCORER, OBJECTS, SUCCESSED, TASK_BY_PAIR, TASKS,
                               TIMEOUT_STOPPED, contacts_key, diagnose, resolve_task, score, task_id, verdict)
from dreamlabs.task_parser import describe, parse_prompt

HAS_SIM = importlib.util.find_spec("dream_examples") is not None and importlib.util.find_spec("mujoco") is not None

# Initial body positions from yam_env.py, in the world frame (m).
INITIAL = {"apple": [0.15, 0.105, 0.792], "banana": [-0.115, 0.085, 0.79], "orange": [0.0, 0.13, 0.794],
           "water_bottle": [0.22, -0.11, 0.845], "bowl": [0.0, -0.04, 0.76], "plate": [-0.22, -0.045, 0.764]}
# Drop offsets (m) that settle in MuJoCo. Dropped dead center, the banana slides off the plate.
DROP_OFFSET = {("banana", "plate"): (0.0, -0.015)}
# Settled height of each body origin above the container origin (m), measured in MuJoCo.
SETTLED_DZ = {("apple", "bowl"): 0.040, ("banana", "bowl"): 0.052, ("orange", "bowl"): 0.036,
              ("water_bottle", "bowl"): 0.086, ("apple", "plate"): 0.035, ("banana", "plate"): 0.026,
              ("orange", "plate"): 0.034, ("water_bottle", "plate"): 0.079}


def frame(t: float, moved: dict | None = None, contacts: dict | None = None) -> dict:
    xyz = {name: list(INITIAL[name]) for name in (*OBJECTS, *CONTAINERS)}
    xyz.update(moved or {})
    touching = {contacts_key(name): ["world"] for name in OBJECTS}
    touching.update({contacts_key(name): list(value) for name, value in (contacts or {}).items()})
    return {"frame": round(t * 20), "sim_seconds": t, "object_xyz_m": xyz, **touching}


def placed_xyz(obj: str, target: str, dx: float = 0.0) -> list[float]:
    x, y, z = INITIAL[target]
    return [x + dx, y, z + SETTLED_DZ[(obj, target)]]


def placed_rows(obj: str, target: str, seconds: float = 1.0, contacts=None, dx: float = 0.0) -> list[dict]:
    on = [target] if contacts is None else contacts
    return [frame(i / 20, {obj: placed_xyz(obj, target, dx)}, {obj: on}) for i in range(round(seconds * 20) + 1)]


def meta(spec) -> dict:
    return {"status": "finished", "task": spec.prompt, "task_id": spec.task_id}


class RegistryTests(unittest.TestCase):
    def test_every_pair_is_registered(self) -> None:
        self.assertEqual(set(TASK_BY_PAIR), {(o, t) for o in OBJECTS for t in CONTAINERS})
        self.assertEqual(len(TASKS), 8)
        self.assertEqual(len({spec.scorer for spec in TASKS.values()}), 8)

    def test_apple_in_bowl_keeps_its_ids(self) -> None:
        spec = TASKS["apple-in-bowl"]
        self.assertEqual((spec.scorer, spec.prompt), ("apple-in-bowl-v1", "put apple into the bowl"))
        self.assertEqual(spec.success_condition,
                         "The released apple stays supported by the bowl and stable for at least 0.5 seconds.")

    def test_task_inferred_from_standard_prompt(self) -> None:
        for spec in TASKS.values():
            with self.subTest(task=spec.task_id):
                self.assertEqual(task_id({"task": spec.prompt.upper()}), spec.task_id)
        self.assertIsNone(task_id({"task": "wave hello"}))


class ScoreTests(unittest.TestCase):
    def test_placed_and_stable_succeeds(self) -> None:
        for spec in TASKS.values():
            with self.subTest(task=spec.task_id):
                result = score(placed_rows(spec.object, spec.target), meta(spec))
                self.assertEqual(result["outcome"], "success", result["reason"])
                self.assertEqual(result["evidence_sim_s"], 0.5)

    def test_wrong_container_fails(self) -> None:
        for spec in TASKS.values():
            other = next(name for name in CONTAINERS if name != spec.target)
            with self.subTest(task=spec.task_id):
                self.assertEqual(score(placed_rows(spec.object, other), meta(spec))["outcome"], "task-failure")

    def test_touching_table_or_arm_fails(self) -> None:
        for spec in TASKS.values():
            for extra in ("world", "left_tip_left", "right_link6"):
                if (spec.task_id, extra) == ("apple-in-bowl", "world"):
                    continue  # v1 is unchanged and has no table rule; its height band already excludes it
                with self.subTest(task=spec.task_id, contact=extra):
                    rows = placed_rows(spec.object, spec.target, contacts=[spec.target, extra])
                    self.assertEqual(score(rows, meta(spec))["outcome"], "task-failure")

    def test_too_short_stable_interval_fails(self) -> None:
        for spec in TASKS.values():
            with self.subTest(task=spec.task_id):
                rows = placed_rows(spec.object, spec.target, seconds=0.4)
                self.assertEqual(score(rows, meta(spec))["outcome"], "task-failure")

    def test_knocked_out_at_the_end_fails(self) -> None:
        for spec in TASKS.values():
            with self.subTest(task=spec.task_id):
                rows = placed_rows(spec.object, spec.target)
                rows.append(frame(1.05, {spec.object: [0.3, 0.3, 0.79]}, {spec.object: ["world"]}))
                self.assertEqual(score(rows, meta(spec))["outcome"], "task-failure")

    def test_never_lifted_reports_it(self) -> None:
        for spec in TASKS.values():
            with self.subTest(task=spec.task_id):
                result = score([frame(i / 20) for i in range(21)], meta(spec))
                self.assertEqual(result["outcome"], "task-failure")
                self.assertIn("never lifted", result["reason"])

    def test_far_outside_container_fails_even_if_contact_reported(self) -> None:
        for spec in TASKS.values():
            if spec.task_id == "apple-in-bowl":
                continue  # v1 uses its own 55 mm radius
            with self.subTest(task=spec.task_id):
                rows = placed_rows(spec.object, spec.target, dx=0.2)
                self.assertEqual(score(rows, meta(spec))["outcome"], "task-failure")

    def test_banana_origin_offset_is_tolerated(self) -> None:
        for target in CONTAINERS:
            spec = TASK_BY_PAIR[("banana", target)]
            with self.subTest(target=target):
                rows = placed_rows("banana", target, dx=0.055)
                self.assertEqual(score(rows, meta(spec))["outcome"], "success")

    def test_old_recording_without_telemetry_is_insufficient(self) -> None:
        spec = TASKS["orange-in-bowl"]
        rows = placed_rows("orange", "bowl")
        for row in rows:
            del row["orange_contacts"]
        self.assertEqual(score(rows, meta(spec))["outcome"], "insufficient-evidence")

    def test_unknown_task_is_unsupported(self) -> None:
        result = score(placed_rows("orange", "bowl"), {"status": "finished", "task_id": "orange-in-cup"})
        self.assertEqual(result["outcome"], "unsupported-task")


class VerdictTests(unittest.TestCase):
    """dreambench_bk/judge_run.py: successed / failed / timeout stopped."""

    def test_mapping(self) -> None:
        cases = {("success", "finished"): SUCCESSED, ("success", "home"): SUCCESSED,
                 ("task-failure", "home"): TIMEOUT_STOPPED, ("task-failure", "finished"): FAILED,
                 ("task-failure", "operator"): FAILED, ("invalid-run", "error"): FAILED,
                 ("insufficient-evidence", None): FAILED}
        for (outcome, stop), expected in cases.items():
            with self.subTest(outcome=outcome, stop=stop):
                self.assertEqual(verdict(outcome, stop), expected)

    def test_words_match_bk(self) -> None:
        self.assertEqual((SUCCESSED, FAILED, TIMEOUT_STOPPED), ("successed", "failed", "timeout stopped"))


def config(prompt: str, candidate: str | None = None, **extra) -> dict:
    variable = "action_limit" if candidate is None else "prompt"
    return {"schema_version": 1, "model": "beta", "seconds": 60, "comparison_variable": variable,
            "baseline": {"prompt": prompt, "action_limit": 48},
            "candidate": {"prompt": candidate or prompt, "action_limit": 24 if candidate is None else 48}, **extra}


class ConfigTests(unittest.TestCase):
    def load(self, data: dict) -> dict:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(data))
            return load_config(path)

    def test_task_scene_and_scorer_come_from_prompt(self) -> None:
        loaded = self.load(config("put the water bottle on the plate"))
        self.assertEqual((loaded["task_id"], loaded["scorer"], loaded["scene_id"], loaded["auto_stop_home_seconds"]),
                         ("water-bottle-on-plate", "water-bottle-on-plate-v1", "yam-tabletop-default", None))
        self.assertEqual(loaded["success_condition"], TASKS["water-bottle-on-plate"].success_condition)

    def test_prompt_comparison_needs_the_same_task(self) -> None:
        self.assertEqual(self.load(config("put apple into the bowl", "carefully place the apple in the bowl"))["task_id"],
                         "apple-in-bowl")
        with self.assertRaisesRegex(ValueError, "same task"):
            self.load(config("put apple into the bowl", "put apple on the plate"))

    def test_explicit_task_still_loads(self) -> None:
        for spec in TASKS.values():
            with self.subTest(task=spec.task_id):
                data = config(spec.prompt, task_id=spec.task_id, scorer=spec.scorer,
                              success_condition=spec.success_condition, scene_id="yam-tabletop-default")
                self.assertEqual(self.load(data)["task_id"], spec.task_id)

    def test_validation_is_idempotent(self) -> None:
        once = validate_config(config("put orange into the bowl", auto_stop_home_seconds=5))
        self.assertEqual(validate_config(json.loads(json.dumps(once))), once)

    def test_rejections(self) -> None:
        good = config("put orange into the bowl")
        for change in ({"scorer": "apple-in-bowl-v1"}, {"scorer": None}, {"success_condition": "anything"},
                       {"task_id": "orange-in-cup"}, {"scene_id": "other"}, {"auto_stop_home_seconds": 0.2},
                       {"auto_stop_home_seconds": 500}, {"auto_stop_home_seconds": True},
                       {"baseline": {"prompt": "wave hello", "action_limit": 48}},
                       {"baseline": {"prompt": "put orange\ninto the bowl", "action_limit": 48}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load({**good, **change})

    def test_home_auto_stop_zero_means_off(self) -> None:
        self.assertIsNone(self.load(config("put apple into the bowl", auto_stop_home_seconds=0))["auto_stop_home_seconds"])
        self.assertEqual(self.load(config("put apple into the bowl", auto_stop_home_seconds=5))["auto_stop_home_seconds"], 5)


class ParsePromptTests(unittest.TestCase):
    def test_every_object_and_container(self) -> None:
        cases = {
            "Please put the apple into the bowl": "apple-in-bowl", "把苹果放进碗里": "apple-in-bowl",
            "put orange into the bowl": "orange-in-bowl", "把橙子放进碗里": "orange-in-bowl",
            "put the banana on the plate": "banana-on-plate", "把香蕉放进碗里": "banana-in-bowl",
            "place the apple on the plate": "apple-on-plate", "把橘子放到盘子上": "orange-on-plate",
            "put the water bottle into the bowl": "water-bottle-in-bowl", "把水瓶放到盘子上": "water-bottle-on-plate",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                spec = parse_prompt(text)
                self.assertEqual((spec.task_id, spec.scorer, spec.tier), (expected, f"{expected}-v1", "calibrated"))
                self.assertIsNone(describe(spec)["note"])

    def test_new_scene_body_gets_generic_scorer(self) -> None:
        scene = Inventory(("apple", "toy_car"), ("bowl", "tray"), "simulator")
        spec = parse_prompt("put the toy car on the tray", scene)
        self.assertEqual((spec.object, spec.target, spec.task_id, spec.scorer, spec.tier),
                         ("toy_car", "tray", "toy-car-on-tray", GENERIC_SCORER, "generic"))
        self.assertIn("Generic placement rule", describe(spec)["note"])

    def test_body_missing_from_scene_is_named_in_error(self) -> None:
        scene = Inventory(("apple",), ("bowl",), "simulator")
        with self.assertRaisesRegex(ValueError, "Name an object the scene has: apple"):
            parse_prompt("put the orange into the bowl", scene)
        with self.assertRaisesRegex(ValueError, "The scene has: bowl"):
            parse_prompt("put the apple on the plate", scene)

    def test_rejects_ambiguous_or_incomplete(self) -> None:
        for text in ("", "put apple and orange into the bowl", "put apple", "apple bowl", "move the cup to the bowl",
                     "put apple into the bowl or plate", "put pineapples into the bowl"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_prompt(text)


def rows_for(obj: str, path: list, target: str = "bowl", target_xyz=(0.0, -0.04, 0.76)) -> list[dict]:
    """One telemetry row per (position, contacts) step, 20 Hz."""
    return [{"frame": i, "sim_seconds": i / 20, contacts_key(obj): list(contacts),
             "object_xyz_m": {obj: list(xyz), target: list(target_xyz)}} for i, (xyz, contacts) in enumerate(path)]


START = [0.3, 0.1, 0.79]
IN_BOWL = [0.0, -0.04, 0.80]


class GenericAndDiagnosticsTests(unittest.TestCase):
    def test_resolve_task(self) -> None:
        spec = resolve_task("toy-car-on-tray")
        self.assertEqual((spec.object, spec.target, spec.scorer, spec.tier), ("toy_car", "tray", GENERIC_SCORER, "generic"))
        self.assertEqual(resolve_task("odd-id", {"object": "red_cube", "target": "plate"}).object, "red_cube")
        self.assertEqual(resolve_task("orange-in-bowl").tier, "calibrated")
        self.assertIsNone(resolve_task("sort-inventory"))

    def test_generic_object_placed_and_stable_succeeds(self) -> None:
        path = [(START, ["world"])] * 5 + [([0.2, 0.0, 0.9], ["left_tip_left"])] * 5 + [(IN_BOWL, ["bowl"])] * 12
        result = score(rows_for("cup", path), {"status": "finished", "task_id": "cup-in-bowl"})
        self.assertEqual((result["outcome"], result["tier"], result["diagnostics"]["stage"]),
                         ("success", "generic", "on-target"))

    def test_generic_object_on_table_fails_with_diagnostics(self) -> None:
        path = [(START, ["world"])] * 5 + [([0.2, 0.0, 0.9], ["left_tip_left"])] * 5 + [([0.2, 0.0, 0.79], ["world"])] * 12
        result = score(rows_for("cup", path), {"status": "finished", "task_id": "cup-in-bowl"})
        self.assertEqual(result["outcome"], "task-failure")
        self.assertEqual((result["diagnostics"]["stage"], result["diagnostics"]["ended_on_table"]),
                         ("lifted-not-placed", True))
        self.assertIn("lifted 110 mm", result["diagnostics"]["summary"])

    def test_object_absent_from_recording_is_unsupported(self) -> None:
        result = score(rows_for("apple", [(START, ["world"])] * 5), {"status": "finished", "task_id": "cup-in-bowl"})
        self.assertEqual(result["outcome"], "unsupported-task")
        self.assertIsNone(result["diagnostics"])

    def test_diagnose_stages(self) -> None:
        stage = lambda path: diagnose(rows_for("orange", path), "orange", "bowl")
        self.assertEqual(stage([(START, ["world"])] * 10)["summary"], "The arms never touched the orange.")
        info = stage([(START, ["world"])] * 3 + [(START, ["world", "right_tip_left"])] * 5 + [(START, ["world"])] * 3)
        self.assertEqual((info["stage"], info["first_arm_contact_sim_s"]), ("touched-not-lifted", 0.15))
        self.assertEqual(stage([(START, ["world"])] * 3 + [([0.4, 0.1, 0.79], ["world"])] * 3)["stage"],
                         "moved-without-arm-contact")
        self.assertIn("also the table",
                      stage([(START, ["world"])] * 3 + [([0.07, -0.04, 0.79], ["bowl", "world"])] * 3)["summary"])


def fake_model(bodies: list[tuple[str, int, int, bool]]):
    """bodies: (name, parent, joint type or -1, collides). Body 0 is the world."""
    names = ["world"] + [b[0] for b in bodies]
    parent = [0] + [b[1] for b in bodies]
    root = [0] * len(names)
    for i in range(1, len(names)):
        root[i] = i if parent[i] == 0 else root[parent[i]]
    joints = [(i + 1, b[2]) for i, b in enumerate(bodies) if b[2] >= 0]
    geoms = [(i + 1, b[3]) for i, b in enumerate(bodies)]
    return SimpleNamespace(
        nbody=len(names), njnt=len(joints), ngeom=len(geoms),
        body=lambda i: SimpleNamespace(name=names[i]),
        body_parentid=list(parent), body_rootid=list(root),
        body_jntnum=list([0] + [1 if b[2] >= 0 else 0 for b in bodies]),
        jnt_bodyid=list([j[0] for j in joints]), jnt_type=list([j[1] for j in joints]),
        geom_bodyid=list([g[0] for g in geoms]),
        geom_contype=list([int(g[1]) for g in geoms]), geom_conaffinity=list([0 for _ in geoms]))


class SceneInventoryTests(unittest.TestCase):
    def test_discovery_rules(self) -> None:
        model = fake_model([("cup", 0, 0, True), ("tray", 0, -1, True), ("arm_base", 0, -1, False),
                            ("arm_link", 3, 3, True), ("top_camera_mount", 0, -1, True), ("decal", 0, -1, False)])
        self.assertEqual(discover(model), Inventory(("cup",), ("tray",), "simulator"))

    @unittest.skipUnless(HAS_SIM, "Dream YAM simulator not installed")
    def test_yam_scene(self) -> None:
        scene = load_inventory()
        self.assertEqual((scene.source, scene.objects, scene.supports),
                         ("simulator", FALLBACK.objects, FALLBACK.supports))

    def test_missing_simulator_package_falls_back(self) -> None:
        def missing(name):
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

        scene_inventory.load_inventory.cache_clear()
        try:
            with mock.patch.object(scene_inventory.importlib.util, "find_spec", missing):
                self.assertEqual(scene_inventory.load_inventory(), FALLBACK)
        finally:
            scene_inventory.load_inventory.cache_clear()


def pose_row(t: float, shoulder: float = 0.0, wrist: float = 0.0) -> dict:
    # Joint order per arm: joint1..joint6, gripper. joint2/joint3 are shoulder and elbow.
    left = [wrist, shoulder, shoulder, wrist, wrist, wrist, 1.0]
    return {"sim_seconds": t,
            "measured_joint_rad_and_gripper": {"left": left, "right": [0.4, 0.0, 0.01, -0.25, -0.3, 0.5, 0.1]}}


def feed(dog: HomeWatchdog, start: float, stop: float, **pose) -> None:
    for index in range(round(start * 20), round(stop * 20) + 1):
        dog.observe(pose_row(index / 20, **pose))


class HomeWatchdogTests(unittest.TestCase):
    def test_initial_hold_does_not_stop(self) -> None:
        dog = HomeWatchdog()
        feed(dog, 0, 20)
        self.assertFalse(dog.should_stop())

    def test_stops_after_more_than_five_seconds_back_home(self) -> None:
        dog = HomeWatchdog(home_seconds=5)
        feed(dog, 0, 1)
        feed(dog, 1.05, 10, shoulder=1.2)
        feed(dog, 10.05, 15.0, shoulder=0.1)  # back home: 4.95 s so far
        self.assertFalse(dog.should_stop())
        dog.observe(pose_row(15.05, shoulder=0.1))
        self.assertTrue(dog.should_stop())
        self.assertEqual(dog.near_since, 10.05)

    def test_leaving_home_again_restarts_dwell(self) -> None:
        dog = HomeWatchdog(home_seconds=5)
        feed(dog, 0, 1)
        feed(dog, 1.05, 5, shoulder=1.0)
        feed(dog, 5.05, 8, shoulder=0.0)
        feed(dog, 8.05, 9, shoulder=0.6)
        feed(dog, 9.05, 13, shoulder=0.0)
        self.assertFalse(dog.should_stop())
        feed(dog, 13.05, 14.1, shoulder=0.0)
        self.assertTrue(dog.should_stop())

    def test_drifting_wrists_do_not_block_home_unless_counted(self) -> None:
        dog = HomeWatchdog(home_seconds=5)
        feed(dog, 0, 1)
        feed(dog, 1.05, 5, shoulder=1.5, wrist=0.9)
        feed(dog, 5.05, 11, shoulder=0.1, wrist=0.5)
        self.assertTrue(dog.should_stop())
        strict = HomeWatchdog(home_seconds=5, joints=(1, 2, 3, 4, 5, 6))
        feed(strict, 0, 1)
        feed(strict, 1.05, 5, shoulder=1.5, wrist=0.9)
        feed(strict, 5.05, 11, shoulder=0.1, wrist=0.5)
        self.assertFalse(strict.should_stop())

    def test_joint_list_and_limits(self) -> None:
        self.assertEqual(joint_list("2,3"), (2, 3))
        self.assertEqual(joint_list("1-6"), (1, 2, 3, 4, 5, 6))
        for text in ("", "0", "7", "a", "2-x"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                joint_list(text)
        with self.assertRaises(ValueError):
            HomeWatchdog(home_seconds=0)


@unittest.skipUnless(HAS_SIM, "Dream YAM simulator not installed")
class SimulatedPlacementTests(unittest.TestCase):
    """Drop each object into its container in the real MuJoCo scene and score the telemetry."""

    @classmethod
    def setUpClass(cls) -> None:
        import mujoco

        spec = importlib.util.find_spec("dream_examples.yam_mujoco")
        sys.path.insert(0, str(next(iter(spec.submodule_search_locations))))
        from yam_env import YamEnv

        cls.mujoco, cls.env = mujoco, YamEnv()

    def contacts(self, name: str) -> list[str]:
        from dreamlabs.trial_recorder import body_contacts

        return body_contacts(self.env.model, self.env.data, name)

    def drop(self, obj: str, target: str, seconds: float = 3.0) -> list[dict]:
        model, data, mujoco = self.env.model, self.env.data, self.mujoco
        self.env.reset()
        joint = lambda name: model.joint("bottle_free" if name == "water_bottle" else f"{name}_free")
        for k, other in enumerate(name for name in OBJECTS if name != obj):
            adr = joint(other).qposadr[0]
            data.qpos[adr: adr + 3] = [0.9 + 0.2 * k, 0.9, 0.9]  # park the rest off the table
        adr = joint(obj).qposadr[0]
        center = data.body(target).xpos.copy()
        dx, dy = DROP_OFFSET.get((obj, target), (0.0, 0.0))
        data.qpos[adr: adr + 3] = [center[0] + dx, center[1] + dy, center[2] + 0.13]
        mujoco.mj_forward(model, data)
        rows = []
        for index in range(round(seconds * 20) + 1):
            rows.append({"frame": index, "sim_seconds": index / 20,
                         **{contacts_key(name): self.contacts(name) for name in OBJECTS},
                         "object_xyz_m": {name: data.body(name).xpos.tolist() for name in (*OBJECTS, *CONTAINERS)}})
            mujoco.mj_step(model, data, nstep=25)  # 500 Hz physics, 20 Hz telemetry
        return rows

    def test_centered_drop_scores_success_and_wrong_container_fails(self) -> None:
        for spec in TASKS.values():
            with self.subTest(task=spec.task_id):
                self.assertEqual(score(self.drop(spec.object, spec.target), meta(spec))["outcome"], "success")
                other = next(name for name in CONTAINERS if name != spec.target)
                self.assertNotEqual(score(self.drop(spec.object, other), meta(spec))["outcome"], "success")


if __name__ == "__main__":
    unittest.main()
