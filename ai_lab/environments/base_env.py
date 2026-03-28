"""Abstract base class for simulation environments."""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseEnv(ABC):
    """Gym-style base environment.

    Sub-classes must implement :meth:`reset`, :meth:`step`, and expose
    ``n_states`` and ``n_actions`` attributes.
    """

    n_states: int
    n_actions: int

    @abstractmethod
    def reset(self) -> int:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Apply *action* and return ``(observation, reward, done, info)``."""

    def render(self) -> Any:  # pragma: no cover
        """Optional human-readable display of the current state."""
