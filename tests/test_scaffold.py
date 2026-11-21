import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call

from dreamlabs.config import Config
from dreamlabs.mocks import CounterSimulation, IncrementPolicy
from dreamlabs.runner import run


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        self.assertEqual(Config.from_env({}), Config())

    def test_overrides(self) -> None:
        env = {"DREAMLABS_STEPS": "5", "DREAMLABS_OUTPUT_DIR": "trial"}
        self.assertEqual(Config.from_env(env), Config(5, Path("trial")))

    def test_invalid_values(self) -> None:
        for value in ("", "zero", "0", "-1"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "DREAMLABS_STEPS"
            ):
                Config.from_env({"DREAMLABS_STEPS": value})
        with self.assertRaisesRegex(ValueError, "DREAMLABS_OUTPUT_DIR"):
            Config.from_env({"DREAMLABS_OUTPUT_DIR": " "})


class RunnerTests(unittest.TestCase):
    def test_mock_loop_resets_and_logs_steps(self) -> None:
        with TemporaryDirectory() as directory:
            config = Config(steps=2, output_dir=Path(directory))
            simulation = CounterSimulation()
            policy = Mock(wraps=IncrementPolicy())
            first_log = run(config, simulation, policy)
            second_log = run(config, simulation, policy)
            entries = [
                json.loads(line)
                for line in first_log.read_text(encoding="utf-8").splitlines()
            ]

        self.assertNotEqual(first_log, second_log)
        self.assertEqual(policy.act.call_args_list, [call(0), call(1)] * 2)
        self.assertEqual(simulation.value, 3)
        self.assertEqual(
            [entry["event"] for entry in entries],
            ["run_started", "step_completed", "step_completed", "run_finished"],
        )
        self.assertEqual(entries[0]["config"], {"steps": 2})
        self.assertEqual([entry["step"] for entry in entries[1:3]], [1, 2])
        self.assertEqual(entries[-1]["completed_steps"], 2)
        for entry in entries:
            self.assertEqual(entry["run_id"], first_log.stem)
            timestamp = datetime.fromisoformat(entry["timestamp"])
            self.assertEqual(timestamp.utcoffset(), timedelta(0))

    def test_failure_is_logged_and_propagated(self) -> None:
        policy = Mock()
        policy.act.side_effect = [1, RuntimeError("mock failure")]
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "mock failure"):
                run(Config(3, output_dir), CounterSimulation(), policy)

            log_path, = output_dir.glob("*.jsonl")
            entries = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [entry["event"] for entry in entries],
            ["run_started", "step_completed", "run_failed"],
        )
        self.assertEqual(entries[-1]["error_type"], "RuntimeError")
        self.assertEqual(entries[-1]["completed_steps"], 1)
