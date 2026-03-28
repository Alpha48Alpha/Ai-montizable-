"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base reinforcement-learning agent interface."""

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Return an action for the given *state*."""

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Update the agent's internal model from a transition."""
