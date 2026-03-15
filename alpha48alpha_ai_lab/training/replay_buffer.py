"""
replay_buffer.py — Experience replay buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and supports
random sampling of mini-batches.
"""

import random
from collections import deque
from typing import List, Tuple

import torch

from alpha48alpha_ai_lab.config import REPLAY_BUFFER_SIZE


Transition = Tuple[int, int, float, int, bool]   # (s, a, r, s', done)


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay.

    Transitions are stored as plain Python tuples and converted to
    PyTorch tensors only when a batch is sampled.
    """

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state:      Current state index.
            action:     Action taken.
            reward:     Reward received.
            next_state: Next state index.
            done:       Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        state_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch and return PyTorch tensors.

        Args:
            batch_size: Number of transitions to sample.
            state_size: Total number of states (for one-hot encoding).

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors.
        """
        batch: List[Transition] = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        def one_hot(indices, size):
            t = torch.zeros(len(indices), size)
            for i, idx in enumerate(indices):
                t[i, idx] = 1.0
            return t

        state_tensor = one_hot(states, state_size)
        next_state_tensor = one_hot(next_states, state_size)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor = torch.tensor(dones, dtype=torch.float32)

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

    def __len__(self) -> int:
        """Return the number of transitions currently stored."""
        return len(self.buffer)
