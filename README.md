# DreamBench

DreamBench is a local regression workbench for Dream-1 developers. Compare a prompt or execution setting in simulation, score the outcome from simulator state, and inspect the recorded evidence before adopting a change.

## Current status

- One real apple-in-bowl comparison ran with identical recorded initial states, Dream-1 Beta, and a 120-second limit. The 48-action baseline failed and the 24-action candidate passed. The result calls for a repeat trial.
- The local Web workbench opens saved comparisons and exports experiment JSON. The CLI accepts prompt or action-limit comparisons for apple-in-bowl. The prompt comparison path was verified with a mocked simulator.
- Both the default tabletop and a fixed apple-offset layout are registered and can run individually. A single experiment can compare both layouts and show per-scene outcomes and replay links. The joint flow was verified with simulated runs. A fresh four-trial experiment is the next validation. Other tasks in the form are drafts until their simulator and scorer are registered.

## Quick start

Open the Web workbench from the repository root:

```sh
python3 web_dashboard.py
```

Visit [the results page](http://127.0.0.1:8765/) or [configure an experiment](http://127.0.0.1:8765/configure). Viewing reports and downloading configurations use no API credits. To start a run in the browser, confirm API usage and select **Run experiment**. The page shows trial progress and links to the report when it finishes. A clean checkout starts with an empty experiment list.

The Web runner uses `~/dream-env/bin/python` by default when available. Use `python3 web_dashboard.py --runner-python /path/to/python` for another environment. To run the same experiment from the CLI, download the JSON and run:

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

## Check the two scene resets offline

```sh
source ~/dream-env/bin/activate
python record_replay.py --scene yam-tabletop-default --check-reset
python record_replay.py --scene yam-tabletop-offset --check-reset
```

The offset layout moves the apple by `[-0.04, 0.025, 0]` metres in MuJoCo
world axes. Both layouts use the same apple-in-bowl scorer. Choose the layout in
the Web configuration page, or use `compare_replays.py --run --scene yam-tabletop-offset`
to run a fresh A/B pair in that layout.

## Compare both YAM layouts

In Configure experiment, select **Both YAM layouts · apple into bowl** and
download the config. The same baseline and candidate run in each scene, for
four trials total. The report lists each matched pair and its two replays.
Only matched pairs with valid task scores contribute to aggregate pass counts.

You can also run the default 48 vs 24 comparison in both layouts directly:

```sh
source ~/dream-env/bin/activate
python compare_replays.py --run --scene yam-tabletop-default --scene yam-tabletop-offset --viewer
```
