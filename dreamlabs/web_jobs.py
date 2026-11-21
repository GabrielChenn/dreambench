"""Run one user-started local comparison and expose its progress."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from threading import Lock, Thread
from uuid import uuid4


class JobConflict(Exception):
    pass


class ExperimentJobs:
    def __init__(self, output_dir: Path, python: Path, script: Path):
        self.output_dir = output_dir
        self.python = python
        self.script = script
        self.lock = Lock()
        self.job: dict | None = None
        self.lines: deque[str] = deque(maxlen=30)
        self.process: subprocess.Popen | None = None
        self.viewport: Path | None = None
        self.closed = False

    def start(self, config: dict, *, viewer: bool, yolo: bool) -> str:
        with self.lock:
            if self.closed or (self.job and self.job["status"] == "running"):
                raise JobConflict("An experiment is already running or the workbench is closing.")
            if not self.python.is_file() or not os.access(self.python, os.X_OK):
                raise ValueError("Dream Labs Python is unavailable. Restart with --runner-python.")
            job_id = uuid4().hex
            self.lines.clear()
            self.viewport = None
            self.job = {
                "id": job_id, "status": "running", "message": "Starting the runner",
                "trials_started": 0, "trials_total": 2 * len(config.get("scene_ids", [config.get("scene_id")])),
            }
        Thread(target=self._run, args=(config, viewer, yolo), daemon=True).start()
        return job_id

    def snapshot(self, job_id: str) -> dict | None:
        with self.lock:
            if self.job is None or self.job["id"] != job_id:
                return None
            return {**self.job, "log": list(self.lines)}

    def viewport_frame(self, job_id: str) -> bytes | None:
        with self.lock:
            path = self.viewport if self.job and self.job["id"] == job_id else None
        if path is None:
            return None
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def _run(self, config: dict, viewer: bool, yolo: bool) -> None:
        report_id = None
        try:
            with tempfile.TemporaryDirectory(prefix="dreambench-job-") as directory:
                path = Path(directory) / "experiment.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                viewport = Path(directory) / "viewport.jpg"
                command = [str(self.python), "-u", str(self.script), "--config", str(path),
                           "--output-dir", str(self.output_dir), "--viewport-jpeg", str(viewport)]
                if viewer:
                    command.append("--viewer")
                if yolo:
                    command.append("--yolo")
                with self.lock:
                    if self.closed:
                        return
                    self.viewport = viewport
                    self.process = subprocess.Popen(command, cwd=self.script.parent, stdin=subprocess.DEVNULL,
                                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                    text=True, errors="replace", start_new_session=True)
                with self.process as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        line = line.strip()
                        if not line:
                            continue
                        with self.lock:
                            self.lines.append(line[:1200])
                            if line.startswith("Trial: "):
                                self.job["trials_started"] += 1
                                self.job["message"] = line.removeprefix("Trial: ")
                            elif line.startswith("Report: "):
                                report = Path(line.removeprefix("Report: ")).resolve()
                                if (report.name == "comparison.json" and report.parent.parent == self.output_dir
                                        and report.parent.name.startswith("comparison-") and report.is_file()):
                                    report_id = report.parent.name
                    code = process.wait()
            with self.lock:
                self.process = None
                if code == 0 and report_id:
                    self.job.update(status="complete", message="Report ready", report_id=report_id)
                else:
                    self.job.update(status="failed", message=f"Runner exited without a report (code {code}).")
        except (OSError, ValueError) as error:
            with self.lock:
                self.process = None
                self.job.update(status="failed", message=str(error))

    def close(self) -> None:
        with self.lock:
            self.closed = True
            process = self.process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
