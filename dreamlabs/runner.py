"""Run a fixed number of steps and record the run lifecycle as JSONL."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from .adapters import Action, Observation, PolicyAdapter, SimulationAdapter
from .config import Config


def _record(log: TextIO, run_id: str, event: str, **fields: Any) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event": event,
        **fields,
    }
    log.write(json.dumps(entry) + "\n")
    log.flush()


def run(
    config: Config,
    simulation: SimulationAdapter[Observation, Action],
    policy: PolicyAdapter[Observation, Action],
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    log_path = config.output_dir / f"{run_id}.jsonl"

    with log_path.open("x", encoding="utf-8") as log:
        _record(log, run_id, "run_started", config={"steps": config.steps})
        completed_steps = 0
        try:
            observation = simulation.reset()
            for _ in range(config.steps):
                action = policy.act(observation)
                observation = simulation.step(action)
                completed_steps += 1
                _record(log, run_id, "step_completed", step=completed_steps)
        except Exception as exc:
            _record(
                log,
                run_id,
                "run_failed",
                completed_steps=completed_steps,
                error_type=type(exc).__name__,
            )
            raise
        _record(log, run_id, "run_finished", completed_steps=completed_steps)
    return log_path
