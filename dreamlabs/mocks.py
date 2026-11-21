"""Mock-only adapters using integers as stand-ins for observations and actions."""


class CounterSimulation:
    def __init__(self) -> None:
        self.value = 0

    def reset(self) -> int:
        self.value = 0
        return self.value

    def step(self, action: int) -> int:
        self.value += action
        return self.value


class IncrementPolicy:
    def act(self, observation: int) -> int:
        return observation + 1
