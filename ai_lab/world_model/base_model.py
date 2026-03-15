"""Abstract base class for world models."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseWorldModel(ABC):
    """Interface for a learned world model.

    A world model predicts the next state and expected reward given a current
    state and action, enabling model-based planning and imagination-based
    training without additional environment interactions.
    """

    @abstractmethod
    def observe(self, state: int, action: int, next_state: int, reward: float) -> None:
        """Record a real transition to update the model."""

    @abstractmethod
    def predict(self, state: int, action: int) -> Tuple[int, float]:
        """Return ``(predicted_next_state, predicted_reward)`` for a state–action pair."""
