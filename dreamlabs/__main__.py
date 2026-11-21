"""Run the offline mock loop with settings from the environment."""

import os
import sys

from .config import Config
from .mocks import CounterSimulation, IncrementPolicy
from .runner import run


def main() -> int:
    try:
        config = Config.from_env(os.environ)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(run(config, CounterSimulation(), IncrementPolicy()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
