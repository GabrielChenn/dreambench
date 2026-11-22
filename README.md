# DreamBench

DreamBench is a local evaluation workbench for Dream-1 users. It runs Dream-1 in Dream YAM Sim (MuJoCo, two YAM arms, three cameras, fruit, a bowl and a plate), judges every trial from simulator ground truth, and shows the evidence: where the policy succeeds, how it fails, and whether a configuration looks safe to try on the real robot.

A learned policy is stochastic and sensitive to what it sees. A single run in the default scene does not tell you whether a prompt, an execution setting or a model is reliable. DreamBench turns a change into a repeatable simulated experiment with recorded, replayable evidence, before anything runs on hardware.

## What it does

| Feature | Summary |
| --- | --- |
| Single runs and A/B comparisons | Run one configuration, or a baseline and a candidate that differ in exactly one field: the prompt, or the number of actions executed per Dream-1 response. |
| Task read from the prompt | An English prompt such as "put orange into the bowl" is judged as `orange-in-bowl`. |
| Ground-truth judging | Calibrated scorers for all eight object/container pairs. Each trial is **successed** or **failed**, with a one-line account of what happened to the object. |
| Random scene layouts | Up to nine seeded random layouts per experiment. Every movable object is moved, overlaps are redrawn before any request, and the report opens with a generalization table. |
| Failure modes | Each trial is classified, for example "grasp failed" or "dropped or placed off target", and modes are counted across layouts. |
| Real-robot safety gate | Each trial gets **pass**, **review** or **block** from its contacts and commanded joint targets. |
| Replay review | A video-style player with a contact timeline, three camera views (optional YOLO boxes), a 3D view of the restored MuJoCo state, and frame-by-frame joint and contact data. |
| Reproducibility | Each experiment saves its configuration, seed, report, trajectory and recordings. A single recorded step can be re-simulated offline. |

Nothing in the results pages, replays, configuration downloads or scene controls calls Dream-1. Only running an experiment uses API credits, and the page asks you to confirm first.

## Quick start

Install the Dream library and Dream YAM Sim as the Dream Labs setup describes, into `~/dream-env`. Then, from the repository root:

```sh
~/dream-env/bin/python web_dashboard.py
```

Open [the results page](http://127.0.0.1:8765/) or [configure an experiment](http://127.0.0.1:8765/configure). The server keeps one headless Dream YAM Sim running and shows its 3D viewport on the configuration page. Drag to rotate, right-drag or shift-drag to pan, and scroll to zoom.

Server options: `--sim-viewer` also opens the desktop MuJoCo window, `--no-sim` starts the simulator only when a page first needs it, `--runner-python PATH` uses another Python environment, and `--port` and `--outputs` change the port and the results folder.

## Configure and run an experiment

On the configuration page:

1. Write the prompt. The page shows the task and scorer it reads from the prompt.
2. Set the model (default `beta`), the time limit per run, **Stop after arms home** and the action limit (1–48 actions executed per response).
3. Optional: tick **Randomize the scene layout to test generalization** and set the number of layouts, the range and the seed.
4. Optional: tick **Compare with a candidate**, choose what changes (prompt or action limit) and set the candidate's value. Both prompts must name the same object and destination.
5. Confirm API credit use and select **Run experiment**. The page shows how many trials will run.

| Control | Effect |
| --- | --- |
| **Run experiment** | Runs the single run, or the baseline and the candidate, in every condition. Each trial starts from a reset scene. |
| **Reset** | Restores objects and arms. Stops a running experiment. |
| **Reset robot** | Drives the arms home over 2 s and leaves the objects. Stops a running experiment. |
| **Stop** | Ends the running trial after the current step, saves and judges it, and skips the remaining trials. |
| **Download config** | Saves the experiment as JSON for the CLI. |

**Stop after arms home** ends a trial once the shoulders and elbows have left their starting pose, come back, and stayed there for the given number of simulated seconds (default 5, 0 turns it off). A trial that ends this way without placing the object is **failed**. Its card shows that it ended with the arms back home.

### Experiment JSON

```json
{
  "schema_version": 1,
  "model": "beta",
  "seconds": 120,
  "auto_stop_home_seconds": 5,
  "comparison_variable": "action_limit",
  "baseline":  {"prompt": "put apple into the bowl", "action_limit": 48},
  "candidate": {"prompt": "put apple into the bowl", "action_limit": 24},
  "random_starts": {"count": 5, "max_offset_m": 0.04, "seed": 1234}
}
```

- Leave out `comparison_variable` and `candidate` for a single run. Use `"prompt"` as the variable to compare prompts.
- `random_starts` is optional.
- `scene_id` (`yam-tabletop-default` or `yam-tabletop-offset`) or `scene_ids` (both) select fixed layouts. The offset layout moves the apple by `[-0.04, 0.025, 0]` m.

### Command line

```sh
source ~/dream-env/bin/activate
python compare_replays.py --config ~/Downloads/dreambench-experiment.json      # an exported experiment
python compare_replays.py --run --random-starts 5 --max-offset 0.04 --seed 1234 # 48 vs 24 actions, random layouts
python compare_replays.py --run --scene yam-tabletop-default --scene yam-tabletop-offset
python compare_replays.py --existing BASELINE_RECORDING CANDIDATE_RECORDING     # re-score, no API calls
```

`--run` accepts `--prompt`, `--model`, `--seconds`, `--baseline-actions`, `--candidate-actions` and `--home-seconds`. `--viewer` opens the MuJoCo window and `--yolo` adds YOLO overlays when the local dependencies and weights are installed. Each experiment writes `outputs/comparison-<id>/` with `experiment.json`, `comparison.json` and one folder per trial. Refresh the results page to see it.

## Judging

The judged object and target come from the prompt. The apple, banana, orange and water bottle in the bowl or on the plate each have a calibrated scorer. Any other object/support pair uses a generic contact rule, and the page says so. A trial **succeeds** when the object finishes supported by the target, clear of the arms and the table, and stable for 0.5 s. Every other trial is **failed**, however it ended: time limit, arms back home, task failure, error or operator stop.

Each trial also gets a diagnosis from the object's trajectory and contacts, for example "The apple was lifted 62 mm at 14.20 s but ended off the bowl, 81 mm from its center on the table."

A comparison decides **retest candidate**, **keep baseline** or **insufficient evidence**. Pairs count only when both trials were scored and started from the same recorded state. A single pair is an observation, so a candidate that passes is marked for a repeat trial rather than adopted.

## Random scene layouts and the generalization table

Each layout gives every movable object (apple, banana, orange, water bottle) its own offset, drawn uniformly within ±`max_offset_m` along MuJoCo world x and y from the seed. The bowl and plate are fixed parts of the scene and do not move.

Before any Dream-1 request, each layout is placed in the simulator. A layout where an object overlaps another body (fruit, bowl, plate, an arm) or leaves the table is redrawn. The report records the rejected draws and their reasons. The same seed and scene give the same layouts. Without `--seed`, the CLI picks a seed and saves it.

The report opens with the **generalization table**:

| Column | Content |
| --- | --- |
| # | The layout number |
| One column per object | Its Δx, Δy in cm. The judged object is marked. |
| Result (or Baseline and Candidate) | The verdict, the failure mode, and the safety level |
| Replay | **Open run replay**, or one link per arm |

The footer counts successes and failures per arm. Each row is a single trial, so read it as an observation, not a success rate.

## Failure modes

Each trial is classified from simulator ground truth:

| Mode | Meaning |
| --- | --- |
| Successed | Placed, released and stable |
| Never reached the object | The arms never touched it |
| Knocked away | It moved without arm contact, for example hit by another object |
| Grasp failed | Touched but never lifted 35 mm |
| Still holding at the end | Lifted, and still in the gripper when the run ended |
| Dropped or placed off target | Lifted, then released away from the target |
| Not released cleanly | Ended on the target but also touching an arm or the table |
| On target, not settled | Ended on the target but not stable long enough |
| Not scored | Not run, invalid start, run error, or no usable recording |

Reports with more than one condition include a **Failure modes** table that counts each mode per arm and lists the layouts in it. A mode describes what happened, not why the model did it.

## Real-robot safety gate

Each trial is checked offline from its recording for behaviour that would be unsafe to repeat on hardware. The check uses the recorded contacts and commanded and measured joints, plus the YAM URDF joint limits from the simulator model. It needs no real robot and makes no API call.

| Level | Rule |
| --- | --- |
| **block** | The arms hit each other, or Dream-1 commanded a joint target outside the URDF limits or a gripper target outside 0–1. The simulator clamps such targets. A real driver may not, and the Dream docs say model actions are not checked against a robot's limits. |
| **review** | An arm hit the table, hit an object or container (other than a fingertip grasp), or touched itself. Open the replay at the linked time. |
| info | Peak measured joint speed and the largest step between consecutive commanded targets. The Dream docs give no YAM speed limit, so these have no threshold. |
| unavailable | The trial was not checked: it was not run, or it was recorded before this check existed or without contact telemetry. |

The report shows the worst level per arm and every rule that fired. Each trial card lists each finding with its count, first time (linked into the replay), peak force and the bodies involved.

The levels are computed automatically. **review** asks a person to look at the replay; the report does not record that someone did. Known limits:

- In this simulator, the folded right arm's link2 and link5 touch in many runs (joints 2 and 3 at 0, joint 4 near −0.23 rad). The collision parts are approximate convex shapes, so this may be an artifact. It is reported as review, not block.
- Contacts are sampled at 20 Hz, so short impacts can be missed. Forces are simulated, not calibrated.
- The joint 2 and 3 lower limits equal the home angle, so nearness to a limit is not used as a rule.

## Motion quality and contacts

Each trial card reports:

- RMS tracking error between commanded and measured joint angles, and the 95th percentile of measured joint jerk. Both use the six joints of each arm, from 20 Hz samples, over the whole recording including warm-up.
- Contact events from MuJoCo contacts sampled at 20 Hz: finger–fruit contact, arm–table and arm–arm collisions. Consecutive sampled frames of one category count as one event, and the first event links to the replay.
- The peak released apple–bowl normal force, in simulated newtons. It is not a calibrated real-world force or a pass/fail threshold.

## Replay review

Open a replay from any trial card or from the generalization table. The page is rebuilt from the trial's `trajectory.jsonl`, so older recordings get the current page.

- **3D review and timeline.** A video-style player (play, pause, speed) over the restored MuJoCo state saved before each control step. Drag to rotate, scroll to zoom, or follow a gripper or object. Contacts are coloured by type with arrows scaled by normal force, and gripper and object paths are drawn. Below it, one lane per contact type: arm–table, arm–arm, arm self-collision, arm hitting an object, finger grasp, object resting on a support. Click a lane or drag to jump. **Previous/Next collision** and the arrow keys step through the recording. The 3D view needs the workbench server and the same MuJoCo version and YAM model as the recording.
- **Camera views.** The three recorded cameras, in step with the timeline. Switch between them, and open full screen (F) with YOLO boxes when YOLO overlays were generated.
- **Frame review.** Both arms' joint targets, measured angles and errors in degrees (errors above 5° highlighted), each object's position, speed and contacts, and every contact before and after the step with its normal force. **Download frame JSON** saves the frame with joint angles in radians.

To re-simulate one recorded step offline and compare it with the recording:

```sh
python record_replay.py --reproduce-frame ~/Downloads/dreambench-frame-438.json
```

## Offline checks

```sh
python record_replay.py --scene yam-tabletop-default --check-reset   # the scene resets to an identical state
python record_replay.py --object-offset apple 0.03 0 --dry-run --headless --seconds 1   # a moved start, no API call
~/dream-env/bin/python -m unittest discover -s tests -v               # tests; none call Dream-1
```

With plain `python3`, the tests that need MuJoCo are skipped.

## Repository map

| Path | Responsibility |
| --- | --- |
| `web_dashboard.py`, `experiment_config.html` | Results pages, configuration page, simulator controls |
| `sim_workspace.py` | The persistent simulator behind the Web workbench: experiments, reset, reset robot, stop |
| `compare_replays.py` | Experiment validation, conditions (scenes × random layouts), trial execution, judging, reports |
| `record_replay.py` | One recorded trial per process for the CLI; reset checks and single-step reproduction |
| `dreamlabs/perturbation.py` | Random layouts: seeded sampling, applying offsets, overlap checks, generalization totals |
| `dreamlabs/failure_modes.py` | Failure-mode classification and counts |
| `dreamlabs/safety.py` | The real-robot safety gate |
| `dreamlabs/scoring.py`, `dreamlabs/task_parser.py`, `dreamlabs/scene_inventory.py` | Scorers, verdicts and diagnoses; prompt-to-task parsing; discovery of movable objects and supports |
| `dreamlabs/trial_recorder.py`, `dreamlabs/contact_events.py` | Synchronized three-camera video, trajectories, physics snapshots and contact sampling |
| `dreamlabs/replay_page.py`, `dreamlabs/review_scene.py` | The replay page and its 3D review |
| `detect_replay.py` | Optional offline YOLO annotations |
| `tests/` | Unit and simulator tests, with Dream-1 replaced by mocks or a scripted policy |
| `outputs/` | Local experiment results, ignored by Git |

## Limitations

- Simulation is not the real robot. Grip friction and object masses are simulator tuning, collision shapes are approximate, and rendered images differ from real cameras.
- Each layout or pair is one trial. Treat a table as observations, and repeat trials before concluding that one configuration is better.
- Every trial calls Dream-1 and uses account credits. Start with a few layouts.
