"""Runtime settings for the offline smoke run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Config:
    steps: int = 3
    output_dir: Path = Path("outputs")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Config":
        raw_steps = env.get("DREAMLABS_STEPS", "3")
        try:
            steps = int(raw_steps)
        except ValueError as exc:
            raise ValueError("DREAMLABS_STEPS must be a positive integer") from exc
        if steps < 1:
            raise ValueError("DREAMLABS_STEPS must be a positive integer")

        output_dir = env.get("DREAMLABS_OUTPUT_DIR", "outputs")
        if not output_dir.strip():
            raise ValueError("DREAMLABS_OUTPUT_DIR must be a nonempty path")
        return cls(steps, Path(output_dir))
