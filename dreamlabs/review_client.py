"""Talk to review_worker.py: one request at a time, with a timeout, restarting the worker if it dies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import time
from threading import Lock
from uuid import uuid4

PREFIX = b"@@review "


class ReviewUnavailable(Exception):
    """The 3D review worker could not answer; the message is shown on the page."""


class ReviewRenderer:
    def __init__(self, python: Path, script: Path, *, timeout: float = 60.0) -> None:
        self.python = python
        self.script = script
        self.timeout = timeout
        self.lock = Lock()
        self.process: subprocess.Popen | None = None
        self.buffer = b""
        self.directory = Path(tempfile.mkdtemp(prefix="dreambench-review-"))

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not self.python.is_file() or not os.access(self.python, os.X_OK):
            raise ReviewUnavailable("Dream Labs Python is unavailable. Restart with --runner-python.")
        try:
            self.process = subprocess.Popen([str(self.python), "-u", str(self.script)], cwd=self.script.parent,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            raise ReviewUnavailable(f"Could not start the 3D review: {error}") from error
        self.buffer = b""
        self._answer()  # the ready line

    def _answer(self) -> dict:
        """Read until one protocol line arrives, or give up after the timeout."""
        process = self.process
        assert process is not None and process.stdout is not None
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    if line.startswith(PREFIX):
                        answer = json.loads(line[len(PREFIX):])
                        if not answer.get("ok"):
                            raise ValueError(answer.get("error") or "The 3D review failed.")
                        return answer
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self._stop()
                    raise ReviewUnavailable("The 3D review did not answer in time.")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    code = process.wait()
                    self.process = None
                    raise ReviewUnavailable(f"The 3D review process exited (code {code}). Try again.")
                self.buffer += chunk

    def _request(self, payload: dict) -> dict:
        with self.lock:
            self._start()
            assert self.process is not None and self.process.stdin is not None
            try:
                self.process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
                self.process.stdin.flush()
            except OSError as error:
                self._stop()
                raise ReviewUnavailable(f"The 3D review process stopped: {error}") from error
            return self._answer()

    def summary(self, recording: Path) -> dict:
        return self._request({"op": "summary", "recording": str(recording)})["summary"]

    def render(self, recording: Path, frame: int, camera: dict, *, contacts: bool, other: bool, trails: bool) -> bytes:
        out = self.directory / f"{uuid4().hex}.jpg"
        try:
            self._request({"op": "render", "recording": str(recording), "frame": frame, "out": str(out),
                           "camera": camera, "contacts": contacts, "other": other, "trails": trails})
            return out.read_bytes()
        finally:
            out.unlink(missing_ok=True)

    def _stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None:
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()

    def close(self) -> None:
        with self.lock:
            self._stop()
        shutil.rmtree(self.directory, ignore_errors=True)
