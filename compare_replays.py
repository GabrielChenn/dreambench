"""Compare two apple-in-bowl recordings, or record both and compare in one command."""

from __future__ import annotations

import argparse
import json
from math import isclose, isfinite
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4

from dreamlabs.trajectory_quality import trajectory_quality
from dreamlabs.contact_events import contact_summary
from dreamlabs.scenarios import DEFAULT_SCENE_ID, SCENES
from dreamlabs.scoring import SCORER_ID, SUCCESS_CONDITION, TASK, TASK_ID, TASKS, score, task_id


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_config(path: Path) -> dict:
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def validate_config(config: object) -> dict:
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Use a schema version 1 experiment configuration from the Web workbench.")
    if "scene_id" in config and "scene_ids" in config:
        raise ValueError("Specify scene_id or scene_ids, not both.")
    scene_ids = config.get("scene_ids", [config.get("scene_id")])
    if (not isinstance(scene_ids, list) or not scene_ids
            or not all(isinstance(name, str) and name in SCENES for name in scene_ids)
            or len(scene_ids) != len(set(scene_ids))):
        raise ValueError("Choose distinct registered scenes.")
    goal_id = config.get("task_id")
    task = TASKS.get(goal_id) if isinstance(goal_id, str) else None
    if (task is None or not all(goal_id in SCENES[name].task_ids for name in scene_ids)
            or config.get("scorer") != task.scorer_id):
        raise ValueError("This scene and scorer pair is a draft or has not been registered.")
    if config.get("success_condition") != task.success_condition:
        raise ValueError("The success condition must match the installed scorer.")
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


def assess(folder: Path) -> dict:
    meta_path = folder / "run.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    track_path = folder / "trajectory.jsonl"
    rows = read_rows(track_path) if track_path.exists() else []
    result = score(rows, metadata)
    result.update({key: metadata.get(key) for key in ("task", "model", "actions", "seconds")})
    result.update(task_id=task_id(metadata), prompt=metadata.get("task"),
                  scene_id=metadata.get("scene_id", DEFAULT_SCENE_ID if task_id(metadata) == TASK_ID else None),
                  scorer=metadata.get("scorer", SCORER_ID if task_id(metadata) == TASK_ID else None),
                  apple_offset_xyz_m=metadata.get("apple_offset_xyz_m"))
    result["initial_state"] = {
        key: rows[0][key] for key in ("object_xyz_m", "measured_joint_rad_and_gripper")
    } if rows else None
    result["motion_quality"] = trajectory_quality(rows)
    result["contact_summary"] = contact_summary(rows)
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
               "--seconds", str(config["seconds"]), "--scene", config["scene_id"]]
    if args.viewport_jpeg:
        command.extend(("--viewport-jpeg", str(args.viewport_jpeg)))
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



def pair_report(baseline: dict, candidate: dict) -> dict:
    decision, reason = compare(baseline, candidate)
    return {"decision": decision, "reason": reason, "baseline": baseline, "candidate": candidate}


def print_pair(pair: dict) -> None:
    for label in ("baseline", "candidate"):
        result = pair[label]
        evidence = result["evidence_sim_s"]
        timestamp = f" at {evidence:.2f} s" if evidence is not None else ""
        print(f"{label.title()}: {result['outcome']}{timestamp}. {result['reason']}")
        if result["replay"]:
            print(f"  Replay: {result['replay']}")
    print(f"{pair['decision']}: {pair['reason']}")


def run_pair(args: argparse.Namespace, folder: Path, config: dict) -> dict:
    results = []
    for label in ("baseline", "candidate"):
        print(f"Trial: {config['scene_id']} · {label}", flush=True)
        try:
            result = assess(run_trial(args, folder, config, label))
        except (OSError, ValueError, KeyError) as error:
            result = {
                "outcome": "run-error", "reason": str(error), "evidence_sim_s": None, "replay": None,
                "actions": config[label]["action_limit"], "prompt": config[label]["prompt"],
            }
        results.append(result)
    pair = pair_report(*results)
    print_pair(pair)
    return pair


def summarize(pairs: list[dict]) -> dict:
    valid = [pair for pair in pairs if pair["decision"] != "insufficient-evidence"]
    wins = sum(pair["baseline"]["outcome"] == "task-failure" and pair["candidate"]["outcome"] == "success"
               for pair in valid)
    regressions = sum(pair["baseline"]["outcome"] == "success" and pair["candidate"]["outcome"] == "task-failure"
                      for pair in valid)
    counts = {label: sum(pair[label]["outcome"] == "success" for pair in valid)
              for label in ("baseline", "candidate")}
    if len(valid) != len(pairs):
        decision, reason = "insufficient-evidence", "At least one scene did not produce a valid matched pair."
    elif regressions:
        decision, reason = "keep-baseline", "Candidate regressed on at least one tested scene."
    elif wins:
        decision, reason = "retest-candidate", "Candidate gained at least one pass without a detected regression. Repeat before adopting."
    else:
        decision, reason = "keep-baseline", "The task outcomes tied across the tested scenes."
    return {"decision": decision, "reason": reason, "valid_pairs": len(valid),
            "planned_pairs": len(pairs), "passes": counts}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="Compare existing recordings without API calls")
    parser.add_argument("--run", action="store_true", help="Record two fresh Dream-1 runs (uses API credits)")
    parser.add_argument("--config", type=Path, help="Run the experiment exported from the Web workbench")
    parser.add_argument("--scene", action="append", choices=tuple(SCENES), help="Scene for --run. Repeat to compare both layouts.")
    parser.add_argument("--model")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--baseline-actions", type=int)
    parser.add_argument("--candidate-actions", type=int)
    parser.add_argument("--viewport-jpeg", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer during each new run")
    parser.add_argument("--yolo", action="store_true", help="Run offline YOLO on each fresh recording")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if sum(bool(mode) for mode in (args.run, args.existing, args.config)) != 1:
        parser.error("Choose --run, --config CONFIG, or --existing BASELINE CANDIDATE.")
    overrides = (args.scene, args.model, args.seconds, args.baseline_actions, args.candidate_actions)
    config = None
    if args.config:
        if any(value is not None for value in overrides):
            parser.error("Put scene, model, time and action limits in the JSON when using --config.")
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
        scene_ids = args.scene or [DEFAULT_SCENE_ID]
        if len(scene_ids) != len(set(scene_ids)):
            parser.error("Choose distinct scenes.")
        scene_fields = {"scene_id": scene_ids[0]} if len(scene_ids) == 1 else {"scene_ids": scene_ids}
        config = {
            "schema_version": 1, **scene_fields, "task_id": TASK_ID, "scorer": SCORER_ID,
            "success_condition": SUCCESS_CONDITION, "model": args.model or "beta", "seconds": seconds,
            "comparison_variable": "action_limit",
            "baseline": {"prompt": TASK, "action_limit": baseline_actions},
            "candidate": {"prompt": TASK, "action_limit": candidate_actions},
        }

    output = (args.output_dir / f"comparison-{uuid4().hex[:12]}").resolve()
    output.mkdir(parents=True)
    if config is None:
        report = pair_report(*map(assess, args.existing))
        print_pair(report)
    else:
        (output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        scene_ids = config.get("scene_ids") or [config["scene_id"]]
        if len(scene_ids) == 1:
            report = run_pair(args, output, {**config, "scene_id": scene_ids[0]})
        else:
            pairs = []
            for scene_id in scene_ids:
                print(f"\nScene: {scene_id}", flush=True)
                scene_folder = output / scene_id
                scene_folder.mkdir()
                pair = run_pair(args, scene_folder, {**config, "scene_id": scene_id})
                pairs.append({"scene_id": scene_id, **pair})
            report = {**summarize(pairs), "scenes": pairs}
            print(f"\n{report['decision']}: {report['reason']}")
        report["experiment"] = config
    path = output / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
