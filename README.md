# DreamBench

DreamBench is a local regression workbench for Dream-1 developers. Compare a prompt or execution setting in simulation, score the outcome from simulator state, and inspect the recorded evidence before adopting a change.

## Current status

- One real apple-in-bowl comparison ran with identical recorded initial states, Dream-1 Beta, and a 120-second limit. The 48-action baseline failed and the 24-action candidate passed. The result calls for a repeat trial.
- The local Web workbench opens saved comparisons and exports experiment JSON. The CLI accepts prompt or action-limit comparisons for apple-in-bowl. The prompt comparison path was verified with a mocked simulator.
- The next milestone is one comparison across two reproducible scenes, with per-scene outcomes and replay links. Other tasks in the form are drafts until their simulator and scorer are registered.

## Quick start

Open the Web workbench from the repository root:

```sh
python3 web_dashboard.py
```

Visit [the results page](http://127.0.0.1:8765/) or [configure an experiment](http://127.0.0.1:8765/configure). The workbench reads local reports and exports configurations without calling Dream-1. A clean checkout starts with an empty experiment list.

To record a fresh comparison, download the JSON configuration and run it from the repository root with the Dream Labs environment:

```sh
source ~/dream-env/bin/activate
python compare_replays.py --config ~/Downloads/dreambench-experiment.json --viewer
```

The command uses Dream-1 credits. Add `--yolo` if the local YOLO dependencies and weights are available. Each run saves its configuration, comparison report, trajectory, and replay under `outputs/`. Refresh the Workbench when the report is ready.

Run the local scaffold tests with `python3 -m unittest discover -s tests -v`.

## Repository map

| Path | Responsibility |
| --- | --- |
| `web_dashboard.py`, `experiment_config.html` | Local results browser and experiment configuration |
| `compare_replays.py` | Experiment validation, paired execution, scoring, and decision report |
| `record_replay.py` | Dream YAM Sim integration, contacts, trajectories, and synchronized three-camera playback |
| `detect_replay.py` | Optional offline YOLO annotations for replay |
| `record_ab.py` | Manual A/B recording for spot checks |
| `dreamlabs/`, `tests/` | Offline scaffold and tests |
| `outputs/` | Local run artifacts, ignored by Git |

Follow the [implementation and acceptance checklist](docs/ROADMAP.md). The [hackathon guide](Dream-1_回归测试台_指导方案.md) describes the project scope, judging criteria, and extended experiment design.
