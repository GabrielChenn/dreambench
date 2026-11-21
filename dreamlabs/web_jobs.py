"""Keep one persistent simulator (sim_workspace.py) for the Web workbench and track its experiments."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from threading import Lock, Thread
from uuid import uuid4

EVENT_PREFIX = "@@sim "


class JobConflict(Exception):
    pass


class SimWorkspace:
    """Start, command and watch the simulator process. Every method is safe to call from any thread.

    state: offline (no process), starting, idle, running (an experiment), resetting, closing.
    """

    def __init__(self, output_dir: Path, python: Path, script: Path, *, viewer: bool = False) -> None:
        self.output_dir = output_dir
        self.python = python
        self.script = script
        self.viewer = viewer
        self.lock = Lock()
        self.process: subprocess.Popen | None = None
        self.state = "offline"
        self.message = "Simulator not started"
        self.notice: str | None = None
        self.job: dict | None = None
        self.lines: deque[str] = deque(maxlen=30)
        self.closed = False
        self.directory = Path(tempfile.mkdtemp(prefix="dreambench-sim-"))
        self.viewport = self.directory / "viewport.jpg"

    # Process management -------------------------------------------------------------------------
    def ensure_started(self) -> None:
        """Start the simulator unless it is already running. Raises ValueError if it cannot start."""
        with self.lock:
            if self.closed:
                raise JobConflict("The workbench is closing.")
            if self.process is not None and self.process.poll() is None:
                return
            if not self.python.is_file() or not os.access(self.python, os.X_OK):
                self.state, self.message = "offline", "Dream Labs Python is unavailable. Restart with --runner-python."
                raise ValueError(self.message)
            command = [str(self.python), "-u", str(self.script), "--output-dir", str(self.output_dir),
                       "--viewport-jpeg", str(self.viewport)]
            if self.viewer:
                command.append("--viewer")
            try:
                process = subprocess.Popen(command, cwd=self.script.parent, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           errors="replace", bufsize=1, start_new_session=True)
            except OSError as error:
                self.state, self.message = "offline", f"Could not start the simulator: {error}"
                raise ValueError(self.message) from error
            self.process = process
            self.state, self.message = "starting", "Starting the simulator"
        Thread(target=self._read, args=(process,), daemon=True).start()

    def _send(self, message: dict) -> None:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None or process.stdin is None:
                raise ValueError("The simulator is not running.")
            try:
                process.stdin.write(json.dumps(message) + "\n")
                process.stdin.flush()
            except OSError as error:
                raise ValueError(f"The simulator stopped accepting commands: {error}") from error

    def _read(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                self._line(line.rstrip("\n"))
        finally:
            code = process.wait()
            for pipe in (process.stdin, process.stdout):
                try:
                    if pipe is not None:
                        pipe.close()
                except OSError:
                    pass
            with self.lock:
                if self.process is process:
                    self.process = None
                    self.state = "offline"
                    self.message = "Simulator closed" if self.closed else f"Simulator exited (code {code}). Reset restarts it."
                    if self.job and self.job["status"] == "running":
                        self.job.update(status="failed", message="The simulator exited during the experiment.")

    def _line(self, line: str) -> None:
        if line.startswith(EVENT_PREFIX):
            try:
                event = json.loads(line[len(EVENT_PREFIX):])
            except ValueError:
                return
            if isinstance(event, dict):
                self._event(event)
            return
        line = line.strip()
        if not line:
            return
        with self.lock:
            self.lines.append(line[:1200])
            job = self.job
            if job and job["status"] == "running" and line.startswith("Trial: "):
                job["trials_started"] += 1
                job["message"] = line.removeprefix("Trial: ")

    def _event(self, event: dict) -> None:
        with self.lock:
            name = event.get("event")
            if name in ("reset_done", "reset_robot_done"):
                # Kept on the idle message that follows, so the page can see the reset finished.
                self.notice = "Scene reset" if name == "reset_done" else "Arms at home"
                self.message = self.notice
            if "state" in event and event["state"] in ("idle", "running", "resetting", "offline"):
                self.state = event["state"]
                if self.state != "idle":
                    self.notice = None
                self.message = {"idle": "Simulator ready" + (f" · {self.notice}" if self.notice else ""),
                                "running": "Experiment running",
                                "resetting": "Resetting the robot" if event.get("command") == "reset_robot"
                                else "Resetting the scene",
                                "offline": event.get("error") or "Simulator unavailable"}[self.state]
            job = self.job
            if name == "experiment_finished" and job and job["id"] == event.get("job_id"):
                report = self._report_id(event.get("report"))
                if report:
                    message = "Stopped · report ready" if event.get("stopped") else "Report ready"
                    job.update(status="stopped" if event.get("stopped") else "complete",
                               message=message, report_id=report)
                else:
                    job.update(status="failed", message=event.get("error") or "The experiment produced no report.")

    def _report_id(self, report: object) -> str | None:
        if not isinstance(report, str):
            return None
        path = Path(report).resolve()
        if (path.name == "comparison.json" and path.parent.parent == self.output_dir.resolve()
                and path.parent.name.startswith("comparison-") and path.is_file()):
            return path.parent.name
        return None

    # Commands from the page ---------------------------------------------------------------------
    def start(self, config: dict, *, yolo: bool) -> str:
        self.ensure_started()
        with self.lock:
            if self.job and self.job["status"] == "running":
                raise JobConflict("An experiment is already running. Stop it first.")
            if self.state == "resetting":
                raise JobConflict("Wait for the reset to finish.")
            job_id = uuid4().hex
            self.lines.clear()
            scenes = config.get("scene_ids") or [config.get("scene_id")]
            self.job = {"id": job_id, "status": "running", "message": "Waiting for the simulator",
                        "trials_started": 0, "trials_total": 2 * len(scenes),
                        "task_id": config.get("task_id")}
        try:
            self._send({"command": "experiment", "job_id": job_id, "config": config, "yolo": yolo})
        except ValueError as error:
            with self.lock:
                self.job.update(status="failed", message=str(error))
            raise
        return job_id

    def command(self, name: str) -> None:
        """Send stop, reset or reset_robot. A reset restarts a simulator that has exited."""
        if name not in ("stop", "reset", "reset_robot"):
            raise ValueError("Unknown simulator command.")
        if name == "stop":
            with self.lock:
                running = self.job is not None and self.job["status"] == "running"
            if not running:
                raise JobConflict("No experiment is running.")
        else:
            self.ensure_started()
        self._send({"command": name})
        with self.lock:
            if name == "stop" and self.job and self.job["status"] == "running":
                self.job["message"] = "Stopping: finishing this step and saving the recording"

    # Status -------------------------------------------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            return {"state": self.state, "message": self.message,
                    "job": {**self.job, "log": list(self.lines)} if self.job else None}

    def snapshot(self, job_id: str) -> dict | None:
        with self.lock:
            if self.job is None or self.job["id"] != job_id:
                return None
            return {**self.job, "log": list(self.lines)}

    def viewport_frame(self) -> bytes | None:
        try:
            return self.viewport.read_bytes()
        except OSError:
            return None

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.state, self.message = "closing", "Simulator closing"
            process = self.process
        if process is not None and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps({"command": "quit"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        shutil.rmtree(self.directory, ignore_errors=True)
