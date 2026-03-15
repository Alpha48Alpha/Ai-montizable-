"""
training/replay_buffer.py — Fixed-capacity experience replay buffer.

The replay buffer stores (state, action, reward, next_state, done)
transitions and allows uniform random sampling of mini-batches.

Uniform replay breaks the temporal correlations inherent in on-policy
data collection, which is a key ingredient of stable DQN training
(Mnih et al., 2015).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity FIFO replay buffer for DQN-style agents.

    When the buffer is full, the oldest transition is automatically
    evicted to make room for the newest one.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store (default: 10 000).
    """

    def __init__(self, capacity: int = 10_000):
        self._buffer: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Add a single transition to the buffer.

        Parameters
        ----------
        state : np.ndarray       — observation before the action.
        action : int             — action taken.
        reward : float           — scalar reward received.
        next_state : np.ndarray  — observation after the action.
        done : bool              — True if the episode ended.
        """
        self._buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample a random mini-batch of transitions (without replacement).

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.  Must be ≤ len(self).

        Returns
        -------
        states      : np.ndarray, shape (batch_size, state_size)
        actions     : np.ndarray, shape (batch_size,)
        rewards     : np.ndarray, shape (batch_size,)
        next_states : np.ndarray, shape (batch_size, state_size)
        dones       : np.ndarray, shape (batch_size,)   — float 0 or 1
        """
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self._buffer.maxlen}, "
            f"current_size={len(self._buffer)})"
        )
