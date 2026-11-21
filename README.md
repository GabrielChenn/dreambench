# DreamBench

Shared workspace for the HackLab Berlin Dream Labs track. The team is still deciding on the project topic.

## Reproduce the simulator baseline

In your existing Dream Labs Python environment, run `dreamcli example yam-mujoco`, then enter `Put apple into the bowl`. The team successfully completed this task in the official simulator.

## Run the local scaffold tests

```sh
python3 -m unittest discover -s tests -v
```

The local scaffold uses mocks. It provides a starting point for the team once the topic is agreed.
