"""SimWorkspace against a fake simulator that speaks sim_workspace.py's protocol. No MuJoCo, no API."""

import json
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dreamlabs.web_jobs import JobConflict, SimWorkspace

# Mirrors sim_workspace.py: JSON commands on stdin, "@@sim {...}" events and log lines on stdout.
FAKE_WORKER = r'''
import argparse, json, sys, threading, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir"); parser.add_argument("--viewport-jpeg"); parser.add_argument("--viewer", action="store_true")
args = parser.parse_args()
def emit(**event): print("@@sim " + json.dumps(event), flush=True)
Path(args.viewport_jpeg).write_bytes(b"\xff\xd8fake-jpeg")
import queue
stop = threading.Event()
commands = queue.Queue()
def read():
    # A reader thread, as in sim_workspace.py, so Stop arrives while an experiment runs.
    for line in sys.stdin:
        message = json.loads(line)
        if message["command"] == "stop":
            stop.set()
        elif message["command"] == "crash":
            import os
            os._exit(3)  # a simulator crash can happen mid-experiment
        else:
            commands.put(message)
    commands.put({"command": "quit"})
threading.Thread(target=read, daemon=True).start()
emit(state="idle")
while True:
    message = commands.get()
    command = message["command"]
    if command == "quit":
        break
    if command in ("reset", "reset_robot"):
        emit(state="resetting", command=command)
        print("  Scene reset." if command == "reset" else "  Arms at home.", flush=True)
        emit(event=command + "_done"); emit(state="idle")
    if command == "experiment":
        stop.clear()
        emit(state="running", job_id=message["job_id"])
        print("Trial: yam-tabletop-default · baseline", flush=True)
        if message["config"].get("hold"):
            while not stop.is_set():
                stop.wait(0.05)
        folder = Path(args.output_dir) / "comparison-fake01"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "comparison.json").write_text("{}")
        print("Report: " + str(folder / "comparison.json"), flush=True)
        emit(event="experiment_finished", job_id=message["job_id"], report=str(folder / "comparison.json"),
             stopped=stop.is_set(), error=None)
        emit(state="idle")
'''


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


class SimWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        script = self.root / "fake_worker.py"
        script.write_text(FAKE_WORKER, encoding="utf-8")
        self.output = self.root / "outputs"
        self.output.mkdir()
        self.sim = SimWorkspace(self.output, Path(sys.executable), script)
        self.addCleanup(self.sim.close)

    def start_idle(self) -> None:
        self.sim.ensure_started()
        wait_for(lambda: self.sim.status()["state"] == "idle")

    def test_starts_idle_and_serves_viewport(self) -> None:
        self.start_idle()
        wait_for(lambda: self.sim.viewport_frame() is not None)
        self.assertTrue(self.sim.viewport_frame().startswith(b"\xff\xd8"))

    def test_experiment_completes_with_report(self) -> None:
        self.start_idle()
        job_id = self.sim.start({"scene_id": "yam-tabletop-default"}, yolo=False)
        wait_for(lambda: self.sim.snapshot(job_id)["status"] != "running")
        job = self.sim.snapshot(job_id)
        self.assertEqual((job["status"], job["report_id"], job["trials_started"]), ("complete", "comparison-fake01", 1))

    def test_second_start_while_running_conflicts_and_stop_ends_it(self) -> None:
        self.start_idle()
        job_id = self.sim.start({"scene_id": "yam-tabletop-default", "hold": True}, yolo=False)
        wait_for(lambda: self.sim.status()["state"] == "running")
        with self.assertRaises(JobConflict):
            self.sim.start({"scene_id": "yam-tabletop-default"}, yolo=False)
        self.sim.command("stop")
        wait_for(lambda: self.sim.snapshot(job_id)["status"] != "running")
        self.assertEqual(self.sim.snapshot(job_id)["status"], "stopped")

    def test_stop_without_experiment_conflicts(self) -> None:
        self.start_idle()
        with self.assertRaises(JobConflict):
            self.sim.command("stop")

    def test_reset_and_reset_robot_report_back(self) -> None:
        self.start_idle()
        self.sim.command("reset")
        wait_for(lambda: "Scene reset" in self.sim.status()["message"])
        self.sim.command("reset_robot")
        wait_for(lambda: "Arms at home" in self.sim.status()["message"])
        self.assertEqual(self.sim.status()["state"], "idle")
        self.assertIn("Arms at home.", self.sim.lines)

    def test_crash_during_experiment_fails_job_and_reset_restarts(self) -> None:
        self.start_idle()
        job_id = self.sim.start({"scene_id": "yam-tabletop-default", "hold": True}, yolo=False)
        wait_for(lambda: self.sim.status()["state"] == "running")
        self.sim._send({"command": "crash"})
        wait_for(lambda: self.sim.status()["state"] == "offline")
        self.assertEqual(self.sim.snapshot(job_id)["status"], "failed")
        self.assertIn("code 3", self.sim.status()["message"])
        self.sim.command("reset")  # restarts the simulator, then resets
        wait_for(lambda: "Scene reset" in self.sim.status()["message"])

    def test_unavailable_python_is_reported(self) -> None:
        sim = SimWorkspace(self.output, self.root / "missing-python", self.root / "fake_worker.py")
        self.addCleanup(sim.close)
        with self.assertRaisesRegex(ValueError, "runner-python"):
            sim.ensure_started()
        self.assertEqual(sim.status()["state"], "offline")

    def test_unknown_command_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.sim.command("teleport")

    def test_report_outside_outputs_is_ignored(self) -> None:
        outside = self.root / "comparison-x" / "comparison.json"
        outside.parent.mkdir()
        outside.write_text("{}")
        self.assertIsNone(self.sim._report_id(str(outside)))
        self.assertIsNone(self.sim._report_id(json.dumps({"no": "path"})))


if __name__ == "__main__":
    unittest.main()
