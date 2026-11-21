"""Record a manually observed Dream YAM Sim A/B experiment."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Optional
from uuid import uuid4


TASK = "Put apple into the bowl"
OUTCOMES = ("success", "task-failure", "timeout", "api-error", "sim-error")


def positive_int(prompt: str, *, optional: bool = False) -> Optional[int]:
    while True:
        value = input(prompt).strip()
        if optional and not value:
            return None
        if value.isdecimal() and int(value) > 0:
            return int(value)
        if optional and value == "0":
            return 0
        print("Enter a positive whole number." if not optional else "Enter a nonnegative whole number or leave blank.")


def markdown_cell(value: object) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", r"\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    model = input("Model ID (leave blank if unknown): ").strip() or None
    layout = input("Initial scene ID [official-default]: ").strip() or "official-default"
    seconds = positive_int("Trial duration in seconds (set the same in both runs): ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid4().hex[:8]}"
    folder = args.output_dir / f"ab-{run_id}"
    folder.mkdir(parents=True, exist_ok=False)
    report = folder / "report.md"
    log = folder / "trials.jsonl"
    report.write_text(
        f"# Apple-in-bowl A/B experiment {run_id}\n\n"
        f"- Task: {TASK}\n"
        f"- Model: {model or 'unknown'}\n"
        f"- Initial scene: {layout}\n"
        f"- Configured duration: {seconds} s\n"
        "- Timing: manually started and stopped; includes operator reaction time.\n"
        "- Success: after release, the apple remains stable inside the bowl.\n\n"
        "| Trial | Max actions per response | Outcome | Elapsed (s) | API calls | End frame | Notes |\n"
        "|---|---:|---|---:|---:|---|---|\n",
        encoding="utf-8",
    )
    print(f"Saving completed trials under {folder.resolve()}")
    for label, actions in (("A baseline", 48), ("B candidate", 24)):
        print(f"\n{label}: in Dream YAM Sim, run /reset, /actions {actions}, /seconds {seconds}.")
        print(f"Prepare this instruction without submitting it yet: {TASK}")
        input("Press Enter here to start timing, then submit the instruction in the simulator. ")
        started = time.monotonic()
        input("Press Enter here when the trial ends or when you observe success. ")
        elapsed = round(time.monotonic() - started, 2)
        print("Use the same stopping rule for both trials.")
        for number, outcome in enumerate(OUTCOMES, start=1):
            print(f"  {number}. {outcome}")
        while True:
            selection = input("Outcome number: ").strip()
            if selection in ("1", "2", "3", "4", "5"):
                break
            print("Choose an outcome from 1 to 5.")
        outcome = OUTCOMES[int(selection) - 1]
        calls = positive_int("API calls (optional): ", optional=True)
        frame = input("End-frame filename or path (optional): ").strip()
        notes = input("Notes or failure description (optional): ").strip()
        trial = {
            "experiment_id": run_id, "trial": label, "task": TASK,
            "model": model, "initial_scene": layout, "duration_seconds": seconds,
            "max_actions_per_response": actions, "outcome": outcome,
            "elapsed_seconds_manual": elapsed, "api_calls": calls,
            "end_frame": frame or None, "notes": notes or None,
        }
        with log.open("a", encoding="utf-8") as output:
            output.write(json.dumps(trial, ensure_ascii=False) + "\n")
        with report.open("a", encoding="utf-8") as output:
            cells = (label, actions, outcome, f"{elapsed:.2f}", calls, frame, notes)
            output.write("| " + " | ".join(map(markdown_cell, cells)) + " |\n")
    print(f"Report: {report.resolve()}\nRaw trials: {log.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nRecording stopped. Completed trials are saved.")
        raise SystemExit(1)
