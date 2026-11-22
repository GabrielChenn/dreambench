# DreamBench

DreamBench is a local regression workbench for Dream-1 developers. Compare a prompt or execution setting in simulation, score the outcome from simulator state, and inspect the recorded evidence before adopting a change.

## Current status

- One real apple-in-bowl comparison ran with identical recorded initial states, Dream-1 Beta, and a 120-second limit. The 48-action baseline failed and the 24-action candidate passed. The result calls for a repeat trial.
- The local Web workbench opens saved comparisons and exports experiment JSON. The CLI accepts prompt or action-limit comparisons for apple-in-bowl. The prompt comparison path was verified with a mocked simulator.
- Both the default tabletop and a fixed apple-offset layout are registered and can run individually. A single experiment can compare both layouts and show per-scene outcomes and replay links. The joint flow was verified with simulated runs. A fresh four-trial experiment is the next validation.
- The judged task is read from the prompt. All eight pairs of the four movable objects (apple, banana, orange, water bottle) and two containers (bowl, plate) have calibrated scorers. Any other pair uses a generic contact rule. Each trial is judged **successed** or **failed**. The page also explains what happened to the object. This judging was ported from the `dreambench_bk` work.
- An experiment can draw up to nine seeded random layouts, each moving every movable object, and report a generalization table. Draws that overlap another body are redrawn before any API call. Verified with a scripted policy in the real MuJoCo scene and dry runs. A Dream-1 grid run is the next validation.
- Every trial gets a failure mode and a real-robot safety level (pass, review or block). Both were checked on existing real recordings, re-scored offline.
- The Web workbench keeps one simulator running, so **Reset**, **Reset robot** and **Stop** work like Dream YAM Sim's `/reset`, `/reset_robot` and stop. These controls and the judging were verified with dry runs and a scripted policy in the real MuJoCo scene, without API calls. A judged Dream-1 run is the next validation.

## Quick start

Open the Web workbench from the repository root:

```sh
~/dream-env/bin/python web_dashboard.py
```

Visit [the results page](http://127.0.0.1:8765/) or [configure an experiment](http://127.0.0.1:8765/configure). The server starts a persistent, headless Dream YAM Sim and shows its 3D viewport on the configuration page. Viewing reports, downloading configurations, and resetting the scene use no API credits. A clean checkout starts with an empty experiment list.

On the configuration page, write the prompt and set the model, time limit, home auto-stop and action limit. That is a single run. To compare, tick **Compare with a candidate**, then choose what changes (prompt or action limit) and set the candidate. The page reads the task from the prompt, in English or Chinese, and shows the scorer it will use. For example, “put orange into the bowl” and “把橙子放进碗里” are both judged as `orange-in-bowl`. When the prompts differ, both must name the same object and destination.

| Control | Effect |
| --- | --- |
| **Run experiment** | After you confirm API usage, runs the single run, or the baseline and the candidate. Each trial starts from a reset scene. |
| **Reset** | Restores the scene: objects and arms. Stops a running experiment. |
| **Reset robot** | Drives the arms home over 2 s and leaves the objects where they are. Stops a running experiment. |
| **Stop** | Stops the running trial after the current step and saves and judges its recording. The remaining trials are not run. |

**Stop after arms home** ends a trial once the shoulders and elbows have left their starting pose, come back, and stayed there for that many simulated seconds. The default is 5. Use 0 to turn it off. A trial that ends this way without placing the object is judged **failed**, and its card shows it ended with the arms back home.

The simulator uses `~/dream-env/bin/python` by default when available. Use `--runner-python /path/to/python` for another environment, `--sim-viewer` to also open the desktop MuJoCo window, or `--no-sim` to start the simulator only when the page first needs it. If the simulator exits, **Reset** starts it again. To run the same experiment from the CLI, download the JSON and run:

```sh
source ~/dream-env/bin/activate
python compare_replays.py --config ~/Downloads/dreambench-experiment.json --viewer
```

The command uses Dream-1 credits. Add `--yolo` if the local YOLO dependencies and weights are available. Each run saves its configuration, comparison report, trajectory, and replay under `outputs/`. Refresh the Workbench when the report is ready.

Run the tests with `~/dream-env/bin/python -m unittest discover -s tests -v`. With plain `python3`, the tests that need MuJoCo are skipped. None of the tests call Dream-1.

## Repository map

| Path | Responsibility |
| --- | --- |
| `web_dashboard.py`, `experiment_config.html` | Local results browser, experiment configuration, and simulator controls |
| `sim_workspace.py` | Persistent simulator for the Web workbench: experiments, reset, reset robot, stop |
| `compare_replays.py` | Experiment validation, task inference, paired execution, judging, and decision report |
| `record_replay.py` | One recorded trial per process for the CLI, and offline reset and frame checks |
| `dreamlabs/scoring.py`, `dreamlabs/task_parser.py`, `dreamlabs/home_watchdog.py` | Scorers and verdicts, prompt-to-task parsing, home auto-stop |
| `dreamlabs/trial_recorder.py` | Contacts, trajectories, and synchronized three-camera recording shared by both runners |
| `detect_replay.py` | Optional offline YOLO annotations for replay |
| `record_ab.py` | Manual A/B recording for spot checks |
| `dreamlabs/`, `tests/` | Offline scaffold and tests |
| `outputs/` | Local run artifacts, ignored by Git |

Follow the [implementation and acceptance checklist](docs/ROADMAP.md). The [hackathon guide](docs/Dream-1_回归测试台_指导方案.md) describes the project scope, judging criteria, and extended experiment design.

## Check the two scene resets offline

```sh
source ~/dream-env/bin/activate
python record_replay.py --scene yam-tabletop-default --check-reset
python record_replay.py --scene yam-tabletop-offset --check-reset
```

The offset layout moves the apple by `[-0.04, 0.025, 0]` metres in MuJoCo
world axes. Both layouts use the same scorers. The Web page runs the default
layout. To use the offset layout, add `"scene_id": "yam-tabletop-offset"` to a
downloaded configuration, or use `compare_replays.py --run --scene yam-tabletop-offset`.

## Compare both YAM layouts

Add `"scene_ids": ["yam-tabletop-default", "yam-tabletop-offset"]` to a
downloaded configuration. The same baseline and candidate run in each scene,
for four trials total. The report lists each matched pair and its two replays.
Only matched pairs with valid task scores contribute to aggregate pass counts.

You can also run the default 48 vs 24 comparison in both layouts directly. Use
`--prompt` to choose the task and `--home-seconds` to change the auto-stop:

```sh
source ~/dream-env/bin/activate
python compare_replays.py --run --scene yam-tabletop-default --scene yam-tabletop-offset --viewer
```

## Generalization: random scene layouts

A learned policy can succeed in the default layout and fail when things are a few centimetres away. To test this, tick **Randomize the scene layout to test generalization** on the configuration page and set the number of layouts (1–9), the range (± cm along MuJoCo world x and y) and the seed. **New seed** draws another set. In JSON the field is `"random_starts": {"count": 5, "max_offset_m": 0.04, "seed": 1234}`. From the CLI:

```sh
python compare_replays.py --run --random-starts 5 --max-offset 0.04 --seed 1234
```

In each layout, every movable object (apple, banana, orange, water bottle) gets its own offset, drawn uniformly from the seed. The bowl and plate stay where they are. Without `--seed` the CLI picks a new seed and saves it in the report. Before any Dream-1 request, each layout is checked in the simulator. A layout where an object is inside another body (fruit, bowl, plate, an arm) or off the table is redrawn. The same seed and scene give the same layouts. Each layout runs the same single run, or the same baseline and candidate, and each trial uses Dream-1 credits.

The report opens with one generalization table. Each row is one layout: every object's Δx, Δy in cm (the judged object is marked), the verdict (**successed** or **failed**), and the replay link in the last column. The footer counts each verdict. The panel also notes how many overlapping draws were redrawn. Each row is a single trial, so read it as an observation, not a success rate.

## Failure modes

Each trial gets a failure mode, read from simulator ground truth (the judged object's contacts and trajectory): **Successed**, **Never reached the object**, **Knocked away**, **Grasp failed**, **Still holding at the end**, **Dropped or placed off target**, **Not released cleanly**, **On target, not settled**, or **Not scored**. The mode appears under each verdict in the generalization table and on each trial card. Reports with more than one condition also get a **Failure modes** table that counts each mode per arm and lists the layouts (#) in it. A mode says what happened, not why the model did it.

## Real-robot safety gate

Each new trial is checked for what would be unsafe to repeat on hardware. The report shows the worst level per arm and every rule that fired:

| Level | Rule |
| --- | --- |
| **block** | The arms hit each other. Dream-1 commanded a joint target outside the YAM URDF limits, or a gripper target outside 0–1. The simulator clamps such targets. A real driver may not, and the Dream docs say model actions are not checked against a robot's limits. |
| **review** | An arm hit the table, an object or container, or itself. |
| info | Peak measured joint speed and the largest step between consecutive commands. The Dream docs give no YAM speed limit, so these have no threshold. |

In this simulator the folded right arm's link2 and link5 touch in many runs (joints 2 and 3 at 0, joint 4 near −0.23 rad). The collision parts are approximate convex shapes, so this may be a collision-model artifact. It is reported as review, not block. Contacts are sampled at 20 Hz, forces are simulated rather than calibrated, and the joint 2 and 3 lower limits equal the home angle, so nearness to a limit is not used. Reports recorded before this check show safety as unavailable. Re-score them with `compare_replays.py --existing BASELINE CANDIDATE`.

## Motion quality in reports

Each trial can report RMS tracking error between the recorded joint target and
measured position, and the 95th percentile of absolute measured joint jerk.
Both use the two arms' six joints, excluding the grippers. Jerk is estimated
from consecutive 20 Hz simulation samples, with gaps excluded. Metrics cover
the full recording, including warm-up, and are descriptive alongside task
outcomes. Missing telemetry is shown as unavailable. These measurements do not
evaluate collision severity or placement force.

## Contact events and placement force

New recordings sample active MuJoCo contacts at 20 Hz. The report separates
finger–fruit contact from arm–table and inter-arm collisions. Consecutive
sampled frames of the same category count as one event. The first event time
links to replay. Older recordings display contact telemetry as unavailable.

The apple–bowl metric is the peak sampled sum of contact-normal magnitudes
when the apple has no detected arm contact. It is reported in newtons in
simulation. A 20 Hz trace can miss short impacts, and this measurement is not
a calibrated real-world force or a pass/fail threshold.

## Review a recording frame by frame

Open a replay from the results page. The page has five parts:

- **Timeline.** A progress bar plus one lane per contact type: arm–table, arm–arm, arm self-collision, arm hitting an object, finger grasp, and object resting on the bowl or plate. Click a lane or drag the bar to jump. **Previous/Next collision** move to the first frame of the nearest collision, and the arrow keys step one frame.
- **Camera views.** The three recorded cameras, kept in step with the timeline.
- **3D review.** The workbench restores each frame from the MuJoCo state saved just before its control step (`physics_before_step`). It renders that state from any angle: drag to rotate, scroll to zoom, or follow a gripper or object. Contacts are colored by type, with arrows scaled by normal force. Gripper and moved-object paths are solid up to the current frame and faint after it. The 3D review needs the workbench server and a recording from the same MuJoCo version and YAM model. Older recordings without snapshots show the other panels only.
- **Frame review.** Both arms' joint targets, measured angles and errors in degrees, with errors above 5° highlighted. Each object's position, speed and contacts. Every contact before the step (from the 3D state) and after it (as recorded), with normal force in newtons.
- **Frame telemetry.** The raw row. Select **Download frame JSON** to save it; the JSON keeps joint angles in radians.

The workbench rebuilds each replay page from its `trajectory.jsonl`, so existing recordings get the current page.

New recordings include the MuJoCo integration state from before the recorded
action. With the same MuJoCo version and YAM model, reproduce it offline:

```sh
source ~/dream-env/bin/activate
python record_replay.py --reproduce-frame ~/Downloads/dreambench-frame-438.json
```

The command re-executes a single simulator step and compares its joint,
gripper, and object positions with the recording. Older recordings can display
and export their joint telemetry. They lack the physics snapshot required for
single-step simulation reproduction.

The configuration page shows the persistent simulator's live 3D viewport,
from the same camera angle as the desktop viewer. Start the server with
`--sim-viewer` if you need the desktop window's native controls.
