# DreamBench

DreamBench is a HackLab Berlin project for Dream‑1 developers who want to know whether a workflow parameter change improves performance in simulation. The first milestone is a repeatable A/B test with automatic scoring and a report that supports a decision about whether to adopt the change.

## Project status

- **Verified in simulation.** The team ran the official Dream Labs YAM/MuJoCo example and completed the apple-in-bowl task.
- **Collaboration ready.** Shichen and Benjamin (`Benjamin0bug`) have access to this repository and can work through pull requests.
- **Next milestone.** Run a baseline and a candidate from the same initial state with one command. Automatic scoring and report generation are planned.

## First milestone

Use the validated apple-in-bowl scene to compare a baseline of up to **48 actions per model response** with a candidate of up to **24**. Keep the model, instruction, initial state, and time limit fixed. Store both configurations so either can be selected again.

The report should show the applied configuration, whether the apple came to rest in the bowl after release, elapsed time, API call count, and an end frame. Record API and simulator errors separately from task outcomes. Verify that a second A/B run starts from the same state, and that switching back to the baseline restores the original setting. A tie is a valid result.

The official example is the starting point for this milestone. The evaluation can also support an existing motion-planning workflow once it exposes reset, parameter setting, single-run execution, and outcome data.

## Collaboration

| Branch | Owner | Scope |
| --- | --- | --- |
| `codex/sim-integration` | Benjamin | Trace and verify reset, the execution parameter, and simulator state access. |
| `codex/benchmark-runner` | Shichen | Validate the success rule, run paired trials, and generate the report. |

Agree on the per-run result format before integrating. Use `docs/execution-path.md` to record source entry points and verification evidence as the investigation progresses. Merge through small pull requests into `main`.

Once the A/B loop works, add a natural-language interface that produces a reviewable configuration diff and calls the same evaluation workflow.

## Reproduce the simulator baseline

In your existing Dream Labs Python environment, run `dreamcli example yam-mujoco`, then enter `Put apple into the bowl`. The team successfully completed this task in the official simulator.

## Run the local scaffold tests

```sh
python3 -m unittest discover -s tests -v
```

The local scaffold uses mocks. It provides the starting point for connecting the real simulator.
