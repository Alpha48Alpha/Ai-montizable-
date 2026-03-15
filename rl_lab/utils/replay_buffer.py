"""Experience replay buffer for DQN-style agents.

Stores ``Transition`` named-tuples and provides uniform random sampling.
The buffer uses a fixed-size circular deque so memory consumption is
bounded at ``capacity`` transitions.

Extension hooks
---------------
- Subclass ``ReplayBuffer`` and override ``sample()`` to implement
  Prioritised Experience Replay (PER).
- Add an n-step return accumulator for multi-step targets.
- Use a ``torch.Tensor``-backed buffer (``TensorReplayBuffer``) for
  faster GPU sampling on large batch sizes.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    """A single environment transition.

    Fields
    ------
    obs : np.ndarray
    action : int | np.ndarray
    reward : float
    next_obs : np.ndarray
    done : bool
    """
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Uniform-random experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample *batch_size* transitions uniformly at random."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
