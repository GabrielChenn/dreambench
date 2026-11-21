"""Compare two apple-in-bowl recordings, or record both and compare in one command."""

from __future__ import annotations

import argparse
import json
from math import dist, isclose, isfinite
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4


TASK = "put apple into the bowl"
TASK_ID = "apple-in-bowl"
SCENE_ID = "yam-tabletop-default"
SCORER_ID = "apple-in-bowl-v1"
SUCCESS_CONDITION = "The released apple stays supported by the bowl and stable for at least 0.5 seconds."
STABLE_SECONDS = 0.50
MAX_STABLE_MOTION_M = 0.008
LIFT_M = 0.035
# Conservative center region for the official bowl's 83 mm outer radius and 58 mm rim.
INSIDE_RADIUS_M = 0.055
MIN_CENTER_HEIGHT_M = 0.020
MAX_CENTER_HEIGHT_M = 0.065


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]



def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Use a schema version 1 experiment configuration from the Web workbench.")
    if (config.get("scene_id"), config.get("task_id"), config.get("scorer")) != (SCENE_ID, TASK_ID, SCORER_ID):
        raise ValueError("This scene and scorer pair is a draft. Only YAM apple-in-bowl is wired to the runner.")
    if config.get("success_condition") != SUCCESS_CONDITION:
        raise ValueError("The apple-in-bowl success condition must match the installed scorer.")
    variable = config.get("comparison_variable")
    if variable not in ("prompt", "action_limit"):
        raise ValueError("Choose prompt or action_limit as the comparison variable.")
    model, seconds = config.get("model"), config.get("seconds")
    if not isinstance(model, str) or not model.strip() or any(char.isspace() for char in model) or len(model) > 100:
        raise ValueError("Use a nonempty model identifier without spaces.")
    if type(seconds) not in (int, float) or not 1 <= seconds <= 600:
        raise ValueError("Use a time limit from 1 to 600 seconds.")
    for name in ("baseline", "candidate"):
        run = config.get(name)
        if not isinstance(run, dict):
            raise ValueError(f"Missing {name} configuration.")
        prompt, limit = run.get("prompt"), run.get("action_limit")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 500:
            raise ValueError(f"Enter a nonempty {name} prompt with no more than 500 characters.")
        if type(limit) is not int or not 1 <= limit <= 48:
            raise ValueError(f"Use an integer {name} action limit from 1 to 48.")
        config[name] = {"prompt": prompt.strip(), "action_limit": limit}
    changed = [field for field in ("prompt", "action_limit")
               if config["baseline"][field] != config["candidate"][field]]
    if changed != [variable]:
        raise ValueError(f"Exactly one field must change: {variable}.")
    return config


def task_id(metadata: dict) -> str | None:
    if "task_id" in metadata:
        return metadata["task_id"]
    return TASK_ID if str(metadata.get("task", "")).strip().lower() == TASK else None

def in_bowl(row: dict) -> bool:
    objects = row["object_xyz_m"]
    apple, bowl = objects["apple"], objects["bowl"]
    contacts = row.get("apple_contacts", [])
    return (
        dist(apple[:2], bowl[:2]) <= INSIDE_RADIUS_M
        and MIN_CENTER_HEIGHT_M <= apple[2] - bowl[2] <= MAX_CENTER_HEIGHT_M
        and "bowl" in contacts
        and not any(name.startswith(("left_", "right_")) for name in contacts)
    )


def score(rows: list[dict], metadata: dict) -> dict:
    if (metadata.get("dry_run") or metadata.get("status") != "finished"
            or metadata.get("recorder_exit_code", 0) != 0 or not rows):
        return {"outcome": "invalid-run", "reason": "No completed task capture to score.", "evidence_sim_s": None}
    if task_id(metadata) != TASK_ID:
        return {"outcome": "unsupported-task", "reason": "The current rule scores only apple-in-bowl.", "evidence_sim_s": None}
    if any("apple_contacts" not in row for row in rows):
        return {"outcome": "insufficient-evidence", "reason": "Recording predates apple contact telemetry.", "evidence_sim_s": None}

    # Use the final stable interval, so an apple later knocked out does not pass.
    final = rows[-1]
    final_apple = final["object_xyz_m"]["apple"]
    start = final["sim_seconds"]
    for row in reversed(rows):
        apple = row["object_xyz_m"]["apple"]
        if not in_bowl(row) or dist(apple, final_apple) > MAX_STABLE_MOTION_M:
            break
        start = row["sim_seconds"]
    if final["sim_seconds"] - start >= STABLE_SECONDS - 1e-6:
        return {"outcome": "success", "reason": "Apple finished supported by the bowl, clear of the arms, and stable.",
                "evidence_sim_s": round(start + STABLE_SECONDS, 3)}

    initial_height = rows[0]["object_xyz_m"]["apple"][2]
    peak = max(rows, key=lambda row: row["object_xyz_m"]["apple"][2])
    if peak["object_xyz_m"]["apple"][2] - initial_height < LIFT_M:
        # Report an observable event, not a claim about the model's internal perception.
        touched = next((row for row in rows if any(
            name.startswith(("left_", "right_")) for name in row["apple_contacts"]
        )), peak)
        return {"outcome": "task-failure", "reason": "Apple never lifted at least 35 mm above its initial height.",
                "evidence_sim_s": touched["sim_seconds"]}
    return {"outcome": "task-failure", "reason": "Apple lifted but did not finish stable and released in the bowl.",
            "evidence_sim_s": final["sim_seconds"]}


def assess(folder: Path) -> dict:
    meta_path = folder / "run.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    track_path = folder / "trajectory.jsonl"
    rows = read_rows(track_path) if track_path.exists() else []
    result = score(rows, metadata)
    result.update({key: metadata.get(key) for key in ("task", "model", "actions", "seconds")})
    result.update(task_id=task_id(metadata), prompt=metadata.get("task"),
                  scene_id=metadata.get("scene_id", SCENE_ID if task_id(metadata) == TASK_ID else None),
                  scorer=metadata.get("scorer", SCORER_ID if task_id(metadata) == TASK_ID else None))
    result["initial_state"] = {
        key: rows[0][key] for key in ("object_xyz_m", "measured_joint_rad_and_gripper")
    } if rows else None
    result["wall_elapsed_s"] = metadata.get("experiment_wall_elapsed_s")
    result["reported_api_requests"] = metadata.get("reported_api_requests")
    yolo = folder / "detections-yolo.jsonl"
    result["yolo_evidence"] = None
    if yolo.exists() and result["evidence_sim_s"] is not None:
        result["yolo_evidence"] = next((
            row for row in read_rows(yolo)
            if row["ran"] and isclose(row["sim_seconds"], result["evidence_sim_s"], abs_tol=0.001)
        ), None)
    replay = folder / ("replay-yolo.html" if (folder / "replay-yolo.html").exists() else "replay.html")
    result["replay"] = str(replay.resolve()) if replay.exists() else None
    return result


def compare(baseline: dict, candidate: dict) -> tuple[str, str]:
    outcomes = baseline["outcome"], candidate["outcome"]
    if any(outcome not in ("success", "task-failure") for outcome in outcomes):
        return "insufficient-evidence", "Both runs need valid task scores."
    for key in ("task_id", "scene_id", "scorer", "model", "seconds", "initial_state"):
        if baseline[key] is None or baseline[key] != candidate[key]:
            return "insufficient-evidence", f"The recordings have missing or different {key}."
    if any(run[key] is None for run in (baseline, candidate) for key in ("prompt", "actions")):
        return "insufficient-evidence", "Missing prompt or action limit."
    changed = sum(baseline[key] != candidate[key] for key in ("prompt", "actions"))
    if changed != 1:
        return "insufficient-evidence", "Exactly one of prompt and action limit must differ."
    if outcomes == ("task-failure", "success"):
        return "retest-candidate", "Candidate passed this pair. Repeat the paired trial before adopting."
    if outcomes == ("success", "task-failure"):
        return "keep-baseline", "Candidate regressed on this pair."
    return "keep-baseline", "The task outcomes tied on this pair."


def run_trial(args: argparse.Namespace, parent: Path, config: dict, label: str) -> Path:
    trial = config[label]
    actions = trial["action_limit"]
    folder = parent / label
    folder.mkdir()
    root = Path(__file__).resolve().parent
    launch = Path(sys.executable).with_name("mjpython") if sys.platform == "darwin" else Path(sys.executable)
    command = [str(launch), str(root / "record_replay.py"), "--output-dir", str(folder),
               f'--task={trial["prompt"]}', f'--model={config["model"]}', "--actions", str(actions),
               "--seconds", str(config["seconds"])]
    if not args.viewer:
        command.append("--headless")
    started = time.monotonic()
    requests = None
    with (folder / "terminal.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                match = re.search(r"· (\d+) requests? ·", line)
                if match:
                    requests = int(match.group(1))
            code = process.wait()
    elapsed = round(time.monotonic() - started, 3)
    recordings = list(folder.glob("replay-*"))
    if not recordings:
        return folder
    recording = recordings[0]
    meta_path = recording / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            task_id=config["task_id"], scene_id=config["scene_id"], scorer=config["scorer"],
            experiment_wall_elapsed_s=elapsed, reported_api_requests=requests, recorder_exit_code=code,
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if code == 0 and args.yolo:
        detector = [sys.executable, str(root / "detect_replay.py"), str(recording)]
        if subprocess.run(detector, check=False).returncode:
            print("YOLO did not complete. The simulation score is still available.", file=sys.stderr)
    return recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="Compare existing recordings without API calls")
    parser.add_argument("--run", action="store_true", help="Record two fresh Dream-1 runs (uses API credits)")
    parser.add_argument("--config", type=Path, help="Run the experiment exported from the Web workbench")
    parser.add_argument("--model")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--baseline-actions", type=int)
    parser.add_argument("--candidate-actions", type=int)
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer during each new run")
    parser.add_argument("--yolo", action="store_true", help="Run offline YOLO on each fresh recording")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if sum(bool(mode) for mode in (args.run, args.existing, args.config)) != 1:
        parser.error("Choose --run, --config CONFIG, or --existing BASELINE CANDIDATE.")
    overrides = (args.model, args.seconds, args.baseline_actions, args.candidate_actions)
    config = None
    if args.config:
        if any(value is not None for value in overrides):
            parser.error("Put model, time and action limits in the JSON when using --config.")
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    elif args.run:
        baseline_actions = 48 if args.baseline_actions is None else args.baseline_actions
        candidate_actions = 24 if args.candidate_actions is None else args.candidate_actions
        seconds = 120 if args.seconds is None else args.seconds
        if (not isfinite(seconds) or seconds <= 0 or not 1 <= baseline_actions <= 48
                or not 1 <= candidate_actions <= 48 or baseline_actions == candidate_actions):
            parser.error("Use a positive duration and two distinct action limits from 1 to 48.")
        config = {
            "schema_version": 1, "scene_id": SCENE_ID, "task_id": TASK_ID, "scorer": SCORER_ID,
            "success_condition": SUCCESS_CONDITION, "model": args.model or "beta", "seconds": seconds,
            "comparison_variable": "action_limit",
            "baseline": {"prompt": TASK, "action_limit": baseline_actions},
            "candidate": {"prompt": TASK, "action_limit": candidate_actions},
        }

    output = (args.output_dir / f"comparison-{uuid4().hex[:12]}").resolve()
    output.mkdir(parents=True)
    if config is not None:
        (output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        folders = tuple(run_trial(args, output, config, label) for label in ("baseline", "candidate"))
    else:
        folders = args.existing
    baseline, candidate = map(assess, folders)
    decision, reason = compare(baseline, candidate)
    report = {"decision": decision, "reason": reason, "baseline": baseline, "candidate": candidate}
    if config is not None:
        report["experiment"] = config
    path = output / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, result in (("Baseline", baseline), ("Candidate", candidate)):
        evidence = result["evidence_sim_s"]
        timestamp = f" at {evidence:.2f} s" if evidence is not None else ""
        print(f"{label}: {result['outcome']}{timestamp}. {result['reason']}")
        if result["replay"]:
            print(f"  Replay: {result['replay']}")
    print(f"{decision}: {reason}\nReport: {path}")


if __name__ == "__main__":
    main()
