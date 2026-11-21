"""Record and replay a Dream YAM Sim run on a shared simulation timeline."""

import argparse
import json
import math
from pathlib import Path
import threading
from uuid import uuid4

from dreamlabs.home_watchdog import DEFAULT_DEPART_RAD, DEFAULT_TOLERANCE_RAD, HomeWatchdog, joint_list
from dreamlabs.replay_page import CAMERAS, write_replay  # noqa: F401 - re-exported for older callers
from dreamlabs.scenarios import DEFAULT_SCENE_ID, SCENES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewport-jpeg", type=Path, help="Publish the latest MuJoCo viewport JPEG")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scene", choices=tuple(SCENES), default=DEFAULT_SCENE_ID)
    parser.add_argument("--reproduce-frame", type=Path, help="Re-run one exported simulation step offline")
    parser.add_argument("--check-reset", action="store_true", help="Check this scene twice without rendering or Dream-1")
    parser.add_argument("--task", default="put apple into the bowl")
    parser.add_argument("--model", default="beta")
    parser.add_argument("--actions", type=int, choices=range(1, 49), default=48, metavar="1–48")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--home-seconds", type=float, default=0,
                        help="Stop once the arms are back home for more than this many sim seconds (0: off)")
    parser.add_argument("--home-joints", type=joint_list, default=(2, 3), help=argparse.SUPPRESS)
    parser.add_argument("--headless", action="store_true", help="Record without opening the MuJoCo viewer")
    parser.add_argument("--dry-run", action="store_true", help="Record simulator warm-up without calling Dream-1")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be a positive finite number.")
    if not math.isfinite(args.home_seconds) or args.home_seconds < 0:
        parser.error("--home-seconds must be 0 (off) or a positive number.")

    try:
        from dreamlabs.trial_recorder import TrialRecorder, ViewportPublisher, load_upstream, recording_env

        dream_control, yam_env = load_upstream()
    except ImportError as error:
        parser.error(f"{error}. Run with ~/dream-env/bin/python.")

    from dreamlabs.frame_restore import reproduce_step

    if args.reproduce_frame:
        env = yam_env.YamEnv()
        try:
            row = json.loads(args.reproduce_frame.read_text(encoding="utf-8"))
            reproduced = reproduce_step(env, row)
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(str(error))
        finally:
            env.close()
        print(json.dumps(reproduced, indent=2))
        if not reproduced["matches_recording"]:
            raise SystemExit(1)
        return

    scene = SCENES[args.scene]
    SceneEnv = recording_env(yam_env.YamEnv)
    SceneEnv.scene = scene

    if args.check_reset:
        env = SceneEnv()
        try:
            def snapshot():
                return {"qpos": env.data.qpos.tolist(), "qvel": env.data.qvel.tolist(),
                        "apple_xyz_m": env.data.body("apple").xpos.tolist(),
                        "bowl_xyz_m": env.data.body("bowl").xpos.tolist()}

            env.reset()
            first = snapshot()
            env.data.joint("apple_free").qpos[0] += 0.01
            state = env.state()
            env.step(state["left"], state["right"])
            env.reset()
            if snapshot() != first:
                raise RuntimeError(f"Scene reset is not repeatable: {args.scene}")
            print(json.dumps({
                "scene_id": args.scene, "apple_offset_xyz_m": scene.apple_offset_xyz_m,
                "checks": 2, "identical_initial_state": True,
                "apple_xyz_m": first["apple_xyz_m"], "bowl_xyz_m": first["bowl_xyz_m"],
            }, indent=2))
        finally:
            env.close()
        return

    output = args.output_dir / f"replay-{uuid4().hex[:12]}"
    env = SceneEnv()
    home = threading.Event()
    watchdog = None
    if args.home_seconds:
        watchdog = HomeWatchdog(args.home_seconds, DEFAULT_TOLERANCE_RAD, DEFAULT_DEPART_RAD, args.home_joints)
    if args.viewport_jpeg is not None:
        env.viewport = ViewportPublisher(args.viewport_jpeg, yam_env.VIEWER_LOOKAT, yam_env.VIEWER_DISTANCE,
                                         yam_env.VIEWER_AZIMUTH, yam_env.VIEWER_ELEVATION)
    status = "interrupted"
    recorder = None
    try:
        recorder = TrialRecorder(output, env.model, watchdog=watchdog, on_home=home.set)
        print(f"Recording to {output.resolve()}", flush=True)
        env.recorder = recorder
        options = [
            f"--task={args.task}", f"--model={args.model}",
            "--actions", str(args.actions), "--seconds", str(args.seconds),
        ]
        if args.headless or args.dry_run:
            options.append("--headless")
        if args.dry_run:
            options.append("--dry-run")
        dream_control.main(options, shared_env=env, stop_event=home if watchdog else None)
        status = "finished"
    finally:
        env.recorder = None
        try:
            if recorder is not None:
                metadata = vars(args).copy()
                for key in ("output_dir", "viewport_jpeg", "reproduce_frame", "check_reset"):
                    metadata.pop(key)
                metadata["home_joints"] = list(args.home_joints)
                metadata.update(status=status, scene_id=args.scene, apple_offset_xyz_m=scene.apple_offset_xyz_m,
                                stop_reason="home" if recorder.home_stop else ("finished" if status == "finished"
                                                                               else "interrupted"))
                recorder.close(metadata)
                if recorder.rows:
                    print(f"Replay: {(output / 'replay.html').resolve()}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
