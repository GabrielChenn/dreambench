"""Compare two DreamBench recordings, or record both and compare in one command.

The judged task comes from the prompt: "put orange into the bowl" is scored as orange-in-bowl. The
eight object/container pairs of the YAM tabletop scene have calibrated scorers; any other pair uses
the generic placement rule. Each trial gets a verdict (successed / failed / timeout stopped) and a
diagnosis of what happened to the object.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from math import isclose
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable
from uuid import uuid4

from dreamlabs.contact_events import contact_summary
from dreamlabs.home_watchdog import DEFAULT_HOME_SECONDS
from dreamlabs.scenarios import DEFAULT_SCENE_ID, SCENES
from dreamlabs.scene_inventory import load_inventory
from dreamlabs.scoring import TASK, resolve_task, score, task_id, verdict
from dreamlabs.task_parser import parse_prompt
from dreamlabs.trajectory_quality import trajectory_quality
from dreamlabs.yolo_status import yolo_unavailable

MAX_HOME_SECONDS = 120.0
TrialRunner = Callable[[Path, dict, str], Path]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_config(path: Path) -> dict:
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def resolve_config_task(config: dict) -> dict:
    """Return the task fields for a configuration: from task_id if given, else from both prompts."""
    goal_id = config.get("task_id")
    if goal_id is not None:
        if not isinstance(goal_id, str):
            raise ValueError("Use a task identifier such as orange-in-bowl.")
        spec = resolve_task(goal_id, config)
        if spec is None:
            raise ValueError(f"Cannot read task {goal_id!r}. Use <object>-in-<support> or <object>-on-<support>.")
    else:
        specs = [parse_prompt(config[name]["prompt"]) for name in trial_labels(config)]
        if len(specs) > 1 and specs[0].task_id != specs[1].task_id:
            raise ValueError(f"Both prompts must ask for the same task: {specs[0].task_id} vs {specs[1].task_id}.")
        spec = specs[0]
    scene = load_inventory()
    if spec.object not in scene.objects or spec.target not in scene.supports:
        raise ValueError(f"The scene has no {spec.object!r} or {spec.target!r}.")
    for key, expected in (("scorer", spec.scorer), ("success_condition", spec.success_condition)):
        if key in config and config[key] != expected:
            raise ValueError(f"The {key.replace('_', ' ')} must match the installed scorer for {spec.task_id}.")
    return {"task_id": spec.task_id, "object": spec.object, "target": spec.target, "scorer": spec.scorer,
            "tier": spec.tier, "success_condition": spec.success_condition}


def trial_labels(config: dict) -> tuple[str, ...]:
    """The trials an experiment records: a baseline/candidate pair, or the baseline alone for a single run."""
    return ("baseline", "candidate") if config.get("comparison_variable") is not None else ("baseline",)


def validate_config(config: object) -> dict:
    """Check an experiment: a comparison (comparison_variable, baseline, candidate) or a single run
    (no comparison_variable, baseline only)."""
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Use a schema version 1 experiment configuration from the Web workbench.")
    if "scene_id" in config and "scene_ids" in config:
        raise ValueError("Specify scene_id or scene_ids, not both.")
    if "scene_id" not in config and "scene_ids" not in config:
        config["scene_id"] = DEFAULT_SCENE_ID
    scene_ids = config.get("scene_ids", [config.get("scene_id")])
    if (not isinstance(scene_ids, list) or not scene_ids
            or not all(isinstance(name, str) and name in SCENES for name in scene_ids)
            or len(scene_ids) != len(set(scene_ids))):
        raise ValueError("Choose distinct registered scenes.")
    variable = config.get("comparison_variable")
    if variable not in (None, "prompt", "action_limit"):
        raise ValueError("Choose prompt or action_limit as the comparison variable, or none for a single run.")
    if variable is None:
        if "candidate" in config:
            raise ValueError("A single run has no candidate. Choose what to compare, or remove the candidate.")
        config["comparison_variable"] = None
    model, seconds = config.get("model"), config.get("seconds")
    if not isinstance(model, str) or not model.strip() or any(char.isspace() for char in model) or len(model) > 100:
        raise ValueError("Use a nonempty model identifier without spaces.")
    if type(seconds) not in (int, float) or not 1 <= seconds <= 600:
        raise ValueError("Use a time limit from 1 to 600 seconds.")
    home = config.get("auto_stop_home_seconds")
    if home is not None and (type(home) not in (int, float) or not (home == 0 or 0.5 <= home <= MAX_HOME_SECONDS)):
        raise ValueError(f"Use 0 (off) or 0.5 to {MAX_HOME_SECONDS:g} seconds for the home auto-stop.")
    config["auto_stop_home_seconds"] = home or None
    for name in trial_labels(config):
        run = config.get(name)
        if not isinstance(run, dict):
            raise ValueError(f"Missing {name} configuration.")
        prompt, limit = run.get("prompt"), run.get("action_limit")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 500:
            raise ValueError(f"Enter a nonempty {name} prompt with no more than 500 characters.")
        if "\n" in prompt or "\r" in prompt:
            raise ValueError(f"Keep the {name} prompt on one line.")
        if type(limit) is not int or not 1 <= limit <= 48:
            raise ValueError(f"Use an integer {name} action limit from 1 to 48.")
        config[name] = {"prompt": prompt.strip(), "action_limit": limit}
    if variable is not None:
        changed = [field for field in ("prompt", "action_limit")
                   if config["baseline"][field] != config["candidate"][field]]
        if changed != [variable]:
            raise ValueError(f"Exactly one field must change: {variable}.")
    config.update(resolve_config_task(config))
    return config


def assess(folder: Path) -> dict:
    meta_path = folder / "run.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    track_path = folder / "trajectory.jsonl"
    rows = read_rows(track_path) if track_path.exists() else []
    result = score(rows, metadata)
    spec = resolve_task(task_id(metadata), metadata)
    result.update({key: metadata.get(key) for key in ("task", "model", "actions", "seconds")})
    result.update(task_id=task_id(metadata), prompt=metadata.get("task"),
                  scene_id=metadata.get("scene_id", DEFAULT_SCENE_ID if spec else None),
                  scorer=metadata.get("scorer", spec.scorer if spec else None),
                  apple_offset_xyz_m=metadata.get("apple_offset_xyz_m"),
                  stop_reason=metadata.get("stop_reason"), home_rule=metadata.get("home_rule"))
    result["verdict"] = verdict(result["outcome"], result["stop_reason"])
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


def trial_metadata(config: dict) -> dict:
    """Fields every trial's run.json carries so it can be scored without the experiment file."""
    return {key: config[key] for key in ("task_id", "object", "target", "scene_id", "scorer")}


def run_trial(args: argparse.Namespace, parent: Path, config: dict, label: str) -> Path:
    """Record one trial in a fresh record_replay.py process."""
    trial = config[label]
    actions = trial["action_limit"]
    folder = parent / label
    folder.mkdir()
    root = Path(__file__).resolve().parent
    launch = Path(sys.executable).with_name("mjpython") if sys.platform == "darwin" else Path(sys.executable)
    command = [str(launch), str(root / "record_replay.py"), "--output-dir", str(folder),
               f'--task={trial["prompt"]}', f'--model={config["model"]}', "--actions", str(actions),
               "--seconds", str(config["seconds"]), "--scene", config["scene_id"]]
    if config.get("auto_stop_home_seconds"):
        command.extend(("--home-seconds", str(config["auto_stop_home_seconds"])))
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
                requests = request_count(line, requests)
            code = process.wait()
    elapsed = round(time.monotonic() - started, 3)
    recordings = list(folder.glob("replay-*"))
    if not recordings:
        return folder
    recording = recordings[0]
    meta_path = recording / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(trial_metadata(config), experiment_wall_elapsed_s=elapsed,
                    reported_api_requests=requests, recorder_exit_code=code)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if code == 0 and args.yolo:
        run_yolo(recording)
    return recording


def request_count(line: str, current: int | None) -> int | None:
    match = re.search(r"· (\d+) requests? ·", line)
    return int(match.group(1)) if match else current


def run_yolo(recording: Path) -> None:
    """Annotate the recording with YOLO, and note in run.json whether it worked so the replay can say why not."""
    error = yolo_unavailable()
    if error is None:
        detector = [sys.executable, str(Path(__file__).resolve().with_name("detect_replay.py")), str(recording)]
        finished = subprocess.run(detector, check=False, capture_output=True, text=True)
        sys.stdout.write(finished.stdout)
        sys.stderr.write(finished.stderr)
        if finished.returncode:
            lines = [line.strip() for line in finished.stderr.splitlines() if line.strip()]
            error = lines[-1] if lines else f"detect_replay.py exited with code {finished.returncode}."
    if error:
        print(f"YOLO did not complete: {error} The simulation score is still available.", file=sys.stderr)
    meta_path = recording / "run.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["yolo"] = {"status": "failed", "error": error} if error else {"status": "done"}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as failure:
        print(f"Could not record the YOLO status in {meta_path}: {failure}", file=sys.stderr)


def pair_report(baseline: dict, candidate: dict) -> dict:
    decision, reason = compare(baseline, candidate)
    return {"decision": decision, "reason": reason, "baseline": baseline, "candidate": candidate}


def single_report(result: dict) -> dict:
    """Report for a run without a candidate: its own verdict, no comparison decision."""
    return {"decision": "single-run", "reason": result["reason"], "baseline": result}


def print_pair(pair: dict) -> None:
    for label in ("baseline", "candidate"):
        if label not in pair:
            continue
        result = pair[label]
        evidence = result["evidence_sim_s"]
        timestamp = f" at {evidence:.2f} s" if evidence is not None else ""
        title = label.title() if "candidate" in pair else "Run"
        print(f"{title}: {result.get('verdict') or result['outcome']}{timestamp}. {result['reason']}")
        if result.get("diagnostics"):
            print(f"  What happened: {result['diagnostics']['summary']}")
        if result["replay"]:
            print(f"  Replay: {result['replay']}")
    if "candidate" in pair:
        print(f"{pair['decision']}: {pair['reason']}")


def unavailable(config: dict, label: str, outcome: str, reason: str) -> dict:
    return {"outcome": outcome, "verdict": None if outcome == "not-run" else "failed", "reason": reason,
            "evidence_sim_s": None, "replay": None,
            "actions": config[label]["action_limit"], "prompt": config[label]["prompt"]}


def run_pair(folder: Path, config: dict, trial_runner: TrialRunner,
             stopped: Callable[[], bool] = lambda: False) -> dict:
    results = []
    for label in trial_labels(config):
        if stopped():
            results.append(unavailable(config, label, "not-run", "The experiment was stopped before this trial."))
            continue
        print(f"Trial: {config['scene_id']} · {label}", flush=True)
        try:
            result = assess(trial_runner(folder, config, label))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            result = unavailable(config, label, "run-error", str(error) or type(error).__name__)
        results.append(result)
    pair = pair_report(*results) if len(results) == 2 else single_report(results[0])
    print_pair(pair)
    return pair


def summarize(pairs: list[dict]) -> dict:
    if not any("candidate" in pair for pair in pairs):  # single runs: count passes, decide nothing
        passed = sum(pair["baseline"]["outcome"] == "success" for pair in pairs)
        return {"decision": "single-run", "reason": f"Passed in {passed} of {len(pairs)} scenes.",
                "planned_runs": len(pairs), "passes": {"baseline": passed}}
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


def run_experiment(config: dict, output: Path, trial_runner: TrialRunner,
                   stopped: Callable[[], bool] = lambda: False) -> Path:
    """Run every scene's baseline/candidate pair (or single run), write comparison.json, and return its path."""
    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    scene_ids = config.get("scene_ids") or [config["scene_id"]]
    if len(scene_ids) == 1:
        report = run_pair(output, {**config, "scene_id": scene_ids[0]}, trial_runner, stopped)
    else:
        pairs = []
        for scene_id in scene_ids:
            print(f"\nScene: {scene_id}", flush=True)
            scene_folder = output / scene_id
            scene_folder.mkdir()
            pair = run_pair(scene_folder, {**config, "scene_id": scene_id}, trial_runner, stopped)
            pairs.append({"scene_id": scene_id, **pair})
        report = {**summarize(pairs), "scenes": pairs}
        print(f"\n{report['decision']}: {report['reason']}")
    if stopped():
        report["stopped"] = True
        report["reason"] = "Stopped by the operator. " + report["reason"]
    report["experiment"] = config
    path = output / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="Compare existing recordings without API calls")
    parser.add_argument("--run", action="store_true", help="Record two fresh Dream-1 runs (uses API credits)")
    parser.add_argument("--config", type=Path, help="Run the experiment exported from the Web workbench")
    parser.add_argument("--scene", action="append", choices=tuple(SCENES), help="Scene for --run. Repeat to compare both layouts.")
    parser.add_argument("--prompt", help=f"Prompt for --run (default: {TASK!r}); it decides the judged task")
    parser.add_argument("--model")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--baseline-actions", type=int)
    parser.add_argument("--candidate-actions", type=int)
    parser.add_argument("--home-seconds", type=float,
                        help=f"Stop a trial once the arms are home for this long (default {DEFAULT_HOME_SECONDS:g}; 0: off)")
    parser.add_argument("--viewport-jpeg", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer during each new run")
    parser.add_argument("--yolo", action="store_true", help="Run offline YOLO on each fresh recording")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if sum(bool(mode) for mode in (args.run, args.existing, args.config)) != 1:
        parser.error("Choose --run, --config CONFIG, or --existing BASELINE CANDIDATE.")
    if args.yolo and (reason := yolo_unavailable()):
        parser.error(f"{reason} Install it or run without --yolo.")
    overrides = (args.scene, args.prompt, args.model, args.seconds, args.baseline_actions,
                 args.candidate_actions, args.home_seconds)
    config = None
    if args.config:
        if any(value is not None for value in overrides):
            parser.error("Put scene, prompt, model, time, action and home limits in the JSON when using --config.")
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    elif args.run:
        baseline_actions = 48 if args.baseline_actions is None else args.baseline_actions
        candidate_actions = 24 if args.candidate_actions is None else args.candidate_actions
        seconds = 120 if args.seconds is None else args.seconds
        scene_ids = args.scene or [DEFAULT_SCENE_ID]
        scene_fields = {"scene_id": scene_ids[0]} if len(scene_ids) == 1 else {"scene_ids": scene_ids}
        prompt = TASK if args.prompt is None else args.prompt
        # Apply the same limits as Web and --config experiments.
        try:
            config = validate_config({
                "schema_version": 1, **scene_fields,
                "model": "beta" if args.model is None else args.model, "seconds": seconds,
                "auto_stop_home_seconds": DEFAULT_HOME_SECONDS if args.home_seconds is None else args.home_seconds,
                "comparison_variable": "action_limit",
                "baseline": {"prompt": prompt, "action_limit": baseline_actions},
                "candidate": {"prompt": prompt, "action_limit": candidate_actions},
            })
        except ValueError as error:
            parser.error(str(error))

    output = (args.output_dir / f"comparison-{uuid4().hex[:12]}").resolve()
    if config is not None:
        run_experiment(config, output, partial(run_trial, args))
        return
    output.mkdir(parents=True)
    report = pair_report(*map(assess, args.existing))
    print_pair(report)
    path = output / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
