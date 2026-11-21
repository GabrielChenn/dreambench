"""Provisional local interfaces. Payload types are chosen by each adapter."""

from typing import Protocol, TypeVar

Observation = TypeVar("Observation")
Action = TypeVar("Action")
ObservationIn = TypeVar("ObservationIn", contravariant=True)
ActionIn = TypeVar("ActionIn", contravariant=True)
ObservationOut = TypeVar("ObservationOut", covariant=True)
ActionOut = TypeVar("ActionOut", covariant=True)


class SimulationAdapter(Protocol[ObservationOut, ActionIn]):
    def reset(self) -> ObservationOut:
        """Initialize a run and return its first observation."""
        ...

    def step(self, action: ActionIn) -> ObservationOut:
        """Apply an action and return the next observation."""
        ...


class PolicyAdapter(Protocol[ObservationIn, ActionOut]):
    def act(self, observation: ObservationIn) -> ActionOut:
        """Choose an action for the given observation."""
        ...
