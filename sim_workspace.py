"""Persistent Dream YAM Sim for the Web workbench: one scene, kept alive between experiments.

web_dashboard.py starts this with the Dream Labs Python and sends one JSON command per stdin line:

    {"command": "experiment", "job_id": "...", "config": {...}, "yolo": false}
    {"command": "stop"}          stop the running task and skip the experiment's remaining trials
    {"command": "reset"}         stop, then restore the scene: objects and arms
    {"command": "reset_robot"}   stop, then drive the arms home over 2 s, leaving the objects
    {"command": "quit"}

These mirror the upstream Dream YAM Sim commands /reset and /reset_robot and its stop (see
dream_examples/yam_mujoco/viewer_workspace.py): a reset first stops the running task. Each trial
still begins from a scene reset, so paired trials start from the same state.

State changes are printed as lines starting with "@@sim " followed by JSON; every other line is log.
The experiment itself prints "Trial: ..." and "Report: ..." lines, as compare_replays.py does.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Callable
from uuid import uuid4

from compare_replays import request_count, run_experiment, run_yolo, trial_metadata, validate_config
from dreamlabs.home_watchdog import DEFAULT_DEPART_RAD, DEFAULT_JOINTS, DEFAULT_TOLERANCE_RAD, HomeWatchdog
from dreamlabs.scenarios import SCENES

EVENT_PREFIX = "@@sim "
STOP_COMMANDS = ("stop", "quit", "reset", "reset_robot")


def emit(**event) -> None:
    print(EVENT_PREFIX + json.dumps(event), flush=True)


class AnyEvent:
    """Look set when any of the given events is set. dream_control.main only calls is_set()."""

    def __init__(self, *events: threading.Event) -> None:
        self.events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self.events)


class RequestCounter:
    """Forward the trial's output and keep the last "· N requests ·" count Dream printed."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.requests: int | None = None
        self.partial = ""

    def write(self, text: str) -> int:
        self.stream.write(text)
        lines = (self.partial + text).split("\n")
        self.partial = lines.pop()
        for line in lines:
            self.requests = request_count(line, self.requests)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()


class Commands:
    """Read stdin on a thread. Stop-like commands set every pending experiment's stop event at once."""

    def __init__(self, stream) -> None:
        self.queue: queue.Queue = queue.Queue()
        self.on_camera: Callable[[dict], None] | None = None  # handled on this thread, even during a trial
        self.tokens: list[threading.Event] = []
        self.lock = threading.Lock()
        self.stream = stream

    def start(self) -> None:
        threading.Thread(target=self.read, daemon=True).start()

    def read(self) -> None:
        for line in self.stream:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if message.get("command") == "camera":
                if self.on_camera is not None:
                    self.on_camera(message)
                continue
            with self.lock:
                if message.get("command") in STOP_COMMANDS:
                    for token in self.tokens:
                        token.set()
                if message.get("command") == "experiment":
                    message["stop_event"] = threading.Event()
                    self.tokens.append(message["stop_event"])
            if message.get("command") != "stop":
                self.queue.put(message)
        self.stop_all()
        self.queue.put({"command": "quit"})

    def stop_all(self) -> None:
        with self.lock:
            for token in self.tokens:
                token.set()

    def done(self, token: threading.Event) -> None:
        with self.lock:
            if token in self.tokens:
                self.tokens.remove(token)

    def next(self, timeout: float) -> dict | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


class Workspace:
    """Own the MuJoCo scene and run one command at a time on the main thread."""

    def __init__(self, args: argparse.Namespace, dream_control, yam_env, recorder_module) -> None:
        self.args = args
        self.dream_control = dream_control
        self.yam_env = yam_env
        self.recorder_module = recorder_module
        self.env = recorder_module.recording_env(yam_env.YamEnv)()
        self.env.viewport = recorder_module.ViewportPublisher(
            args.viewport_jpeg, yam_env.VIEWER_LOOKAT, yam_env.VIEWER_DISTANCE,
            yam_env.VIEWER_AZIMUTH, yam_env.VIEWER_ELEVATION) if args.viewport_jpeg else None
        self.viewer = None

    def set_camera(self, message: dict) -> None:
        """Move the Web viewport camera ({"reset": true} restores the default). Called from the stdin thread."""
        if self.env.viewport is None:
            return
        try:
            self.env.viewport.set_camera(None if message.get("reset") else message.get("view"))
        except ValueError as error:
            print(f"  Ignored viewport camera: {error}", flush=True)

    # Keep the viewer and Web viewport alive between commands.
    def refresh(self, *, force: bool = False) -> None:
        if self.env.viewport is not None:
            self.env.viewport.publish(self.env, force=force)
        if self.viewer is not None:
            self.viewer.sync()

    def idle(self) -> None:
        import mujoco

        mujoco.mj_step(self.env.model, self.env.data, nstep=10)
        self.refresh()

    def reset(self) -> None:
        self.env.reset()
        self.refresh(force=True)
        print("  Scene reset.", flush=True)

    def reset_robot(self) -> None:
        # Drive the arms home through the position servos; the objects stay where they are.
        for left, right in self.yam_env.home_path(self.env.state()):
            tick = time.monotonic()
            self.env.step(left, right)
            self.refresh()
            time.sleep(max(0, 0.05 - (time.monotonic() - tick)))
        self.refresh(force=True)
        print("  Arms at home.", flush=True)

    def trial(self, stop: threading.Event, parent: Path, config: dict, label: str) -> Path:
        """Record one trial in this scene: reset it, run Dream-1, judge from what was recorded."""
        trial = config[label]
        folder = parent / label
        folder.mkdir()
        self.env.scene = SCENES[config["scene_id"]]
        self.env.reset()
        home = threading.Event()
        watchdog = None
        if config.get("auto_stop_home_seconds"):
            watchdog = HomeWatchdog(config["auto_stop_home_seconds"], DEFAULT_TOLERANCE_RAD,
                                    DEFAULT_DEPART_RAD, DEFAULT_JOINTS)
        recording = folder / f"replay-{uuid4().hex[:12]}"
        recorder = self.recorder_module.TrialRecorder(recording, self.env.model, watchdog=watchdog, on_home=home.set)
        print(f"Recording to {recording}", flush=True)
        options = [f"--task={trial['prompt']}", f"--model={config['model']}",
                   "--actions", str(trial["action_limit"]), "--seconds", str(config["seconds"]), "--quiet"]
        if self.viewer is None:
            options.append("--headless")
        if self.args.dry_run:
            options.append("--dry-run")
        counter = RequestCounter(sys.stdout)
        status, code, started = "interrupted", 0, time.monotonic()
        self.env.recorder = recorder
        try:
            with redirect_stdout(counter):
                self.dream_control.main(options, shared_env=self.env, shared_viewer=self.viewer,
                                        stop_event=AnyEvent(stop, home))
            status = "finished"
        except (Exception, SystemExit) as error:  # a failed trial must not end the workspace
            code = 1
            print(f"  Trial stopped: {error}", flush=True)
        finally:
            self.env.recorder = None
            self.env.hold()
            stop_reason = ("home" if recorder.home_stop else "operator" if stop.is_set()
                           else "finished" if status == "finished" else "error")
            metadata = {
                "task": trial["prompt"], "model": config["model"], "actions": trial["action_limit"],
                "seconds": config["seconds"], "headless": self.viewer is None, "dry_run": self.args.dry_run,
                "scene": config["scene_id"], "status": status, "stop_reason": stop_reason,
                "apple_offset_xyz_m": self.env.scene.apple_offset_xyz_m, **trial_metadata(config),
                "experiment_wall_elapsed_s": round(time.monotonic() - started, 3),
                "reported_api_requests": counter.requests, "recorder_exit_code": code,
                "persistent_workspace": True,
            }
            recorder.close(metadata)
        if code == 0 and config.get("yolo"):
            run_yolo(recording)
        return recording

    def experiment(self, message: dict) -> None:
        job_id, stop = message.get("job_id"), message["stop_event"]
        emit(state="running", job_id=job_id)
        report = None
        error = None
        try:
            config = validate_config(message.get("config"))
            config["yolo"] = message.get("yolo") is True
            output = (Path(self.args.output_dir) / f"comparison-{uuid4().hex[:12]}").resolve()
            report = run_experiment(config, output, lambda *trial: self.trial(stop, *trial), stop.is_set)
        except Exception as failure:  # report it to the page; keep the scene alive
            error = str(failure) or type(failure).__name__
            print(f"  Experiment failed: {error}", flush=True)
        emit(event="experiment_finished", job_id=job_id, report=str(report) if report else None,
             stopped=stop.is_set(), error=error)

    def run(self, commands: Commands) -> None:
        emit(state="idle")
        while self.viewer is None or self.viewer.is_running():
            message = commands.next(timeout=0.02)
            if message is None:
                self.idle()
                continue
            command = message.get("command")
            if command == "quit":
                break
            if command in ("reset", "reset_robot"):
                emit(state="resetting", command=command)
                (self.reset if command == "reset" else self.reset_robot)()
                emit(event=f"{command}_done")
            elif command == "experiment":
                try:
                    if message["stop_event"].is_set():
                        emit(event="experiment_finished", job_id=message.get("job_id"), report=None,
                             stopped=True, error=None)
                    else:
                        self.experiment(message)
                finally:
                    commands.done(message["stop_event"])
            emit(state="idle")

    def close(self) -> None:
        self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--viewport-jpeg", type=Path, help="Keep the latest 3D viewport JPEG here")
    parser.add_argument("--viewer", action="store_true", help="Also open the desktop MuJoCo viewer")
    parser.add_argument("--dry-run", action="store_true", help="Run trials without calling Dream-1 (warm-up only)")
    args = parser.parse_args()
    try:
        from dreamlabs import trial_recorder

        dream_control, yam_env = trial_recorder.load_upstream()
    except ImportError as error:
        print(f"  Simulator unavailable: {error}", flush=True)
        emit(state="offline", error=str(error))
        raise SystemExit(1)
    commands = Commands(sys.stdin)
    commands.start()
    workspace = Workspace(args, dream_control, yam_env, trial_recorder)
    if workspace.env.viewport is not None:
        commands.on_camera = workspace.set_camera
        emit(camera=workspace.env.viewport.default)  # the page starts from, and resets to, this view
    try:
        if args.viewer:
            import mujoco.viewer

            context = mujoco.viewer.launch_passive(workspace.env.model, workspace.env.data,
                                                   show_left_ui=False, show_right_ui=False)
        else:
            context = nullcontext(None)
        with context as viewer:
            if viewer is not None:
                yam_env.aim_viewer(viewer)
            workspace.viewer = viewer
            workspace.refresh(force=True)
            workspace.run(commands)
    finally:
        workspace.close()


if __name__ == "__main__":
    main()
