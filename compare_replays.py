"""Compare two apple-in-bowl recordings, or record both and compare in one command."""

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
STABLE_SECONDS = 0.50
MAX_STABLE_MOTION_M = 0.008
LIFT_M = 0.035
# Conservative center region for the official bowl's 83 mm outer radius and 58 mm rim.
INSIDE_RADIUS_M = 0.055
MIN_CENTER_HEIGHT_M = 0.020
MAX_CENTER_HEIGHT_M = 0.065


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
    if metadata.get("task", "").strip().lower() != TASK:
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
    for key in ("task", "model", "seconds", "initial_state"):
        if baseline[key] is None or baseline[key] != candidate[key]:
            return "insufficient-evidence", f"The recordings have missing or different {key}."
    if baseline["actions"] is None or candidate["actions"] is None or baseline["actions"] == candidate["actions"]:
        return "insufficient-evidence", "Use two distinct recorded action limits."
    if outcomes == ("task-failure", "success"):
        return "retest-candidate", "Candidate passed this pair. Repeat the paired trial before adopting."
    if outcomes == ("success", "task-failure"):
        return "keep-baseline", "Candidate regressed on this pair."
    return "keep-baseline", "The task outcomes tied on this pair."


def run_trial(args: argparse.Namespace, parent: Path, actions: int) -> Path:
    folder = parent / f"actions-{actions}"
    folder.mkdir()
    root = Path(__file__).resolve().parent
    launch = Path(sys.executable).with_name("mjpython") if sys.platform == "darwin" else Path(sys.executable)
    command = [str(launch), str(root / "record_replay.py"), "--output-dir", str(folder),
               "--task", TASK, "--model", args.model, "--actions", str(actions),
               "--seconds", str(args.seconds)]
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
        meta.update(experiment_wall_elapsed_s=elapsed, reported_api_requests=requests, recorder_exit_code=code)
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
    parser.add_argument("--model", default="beta")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--baseline-actions", type=int, default=48)
    parser.add_argument("--candidate-actions", type=int, default=24)
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer during each new run")
    parser.add_argument("--yolo", action="store_true", help="Run offline YOLO on each fresh recording")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.run == bool(args.existing):
        parser.error("Choose either --run or --existing BASELINE CANDIDATE.")
    if args.run and (not isfinite(args.seconds) or args.seconds <= 0
                     or not 1 <= args.baseline_actions <= 48 or not 1 <= args.candidate_actions <= 48
                     or args.baseline_actions == args.candidate_actions):
        parser.error("Use a positive duration and two distinct action limits from 1 to 48.")
    output = (args.output_dir / f"comparison-{uuid4().hex[:12]}").resolve()
    output.mkdir(parents=True)
    folders = (run_trial(args, output, args.baseline_actions), run_trial(args, output, args.candidate_actions)
               ) if args.run else args.existing
    baseline, candidate = map(assess, folders)
    decision, reason = compare(baseline, candidate)
    report = {"decision": decision, "reason": reason, "baseline": baseline, "candidate": candidate}
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
