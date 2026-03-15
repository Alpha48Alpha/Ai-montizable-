"""Deep Q-Network (DQN) agent.

Implements DQN (Mnih et al., 2015) with the following improvements:
  - **Double DQN** (van Hasselt et al., 2016): decouples action selection
    from value estimation to reduce over-estimation bias
  - **Experience replay buffer** (uniform random sampling)
  - **Target network** with periodic hard-update (``target_update_freq``)
  - Epsilon-greedy exploration with linear annealing
  - Gradient clipping for training stability

This agent works exclusively with **discrete** action spaces.

Extension hooks
---------------
- Swap ``MLP`` for a ``CNNEncoder`` backbone for pixel observations
- Add prioritised experience replay (PER) via a custom ReplayBuffer subclass
- Add a distributional head (C51 / QR-DQN) for better value estimation
- Combine with a world model for Dreamer-style planning
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_lab.agents.base import BaseAgent
from rl_lab.models.mlp import MLP
from rl_lab.utils.replay_buffer import ReplayBuffer, Transition


class DQNAgent(BaseAgent):
    """Double DQN agent for discrete action spaces.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
        Number of discrete actions.
    hidden_sizes : list[int]
    lr : float
    gamma : float
    buffer_capacity : int
        Maximum number of transitions stored in the replay buffer.
    batch_size : int
    target_update_freq : int
        Number of gradient steps between hard target-network updates.
    eps_start : float
        Initial epsilon for ε-greedy exploration.
    eps_end : float
        Final (minimum) epsilon.
    eps_decay_steps : int
        Linear annealing duration (steps).
    grad_clip : float | None
    device : str
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int] | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10_000,
        grad_clip: float | None = 10.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_dim, act_dim, device)
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        # Q-networks
        self.q_net = MLP(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self._step_count: int = 0

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def epsilon(self) -> float:
        """Current exploration probability (linearly decayed)."""
        fraction = min(self._step_count / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    # ------------------------------------------------------------------ #
    #  BaseAgent API                                                       #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.act_dim))
        obs_t = self.preprocess_obs(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        """Push one transition into the replay buffer."""
        self.replay_buffer.push(
            Transition(
                obs=obs.astype(np.float32),
                action=action,
                reward=reward,
                next_obs=next_obs.astype(np.float32),
                done=terminated,
            )
        )

    def update(self) -> dict:  # type: ignore[override]
        """Sample a mini-batch and perform one Double-DQN gradient step."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)

        obs_t = torch.tensor(np.stack([t.obs for t in batch]),
                             dtype=torch.float32, device=self.device)
        act_t = torch.tensor([t.action for t in batch],
                              dtype=torch.long, device=self.device)
        rew_t = torch.tensor([t.reward for t in batch],
                              dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(np.stack([t.next_obs for t in batch]),
                                  dtype=torch.float32, device=self.device)
        done_t = torch.tensor([t.done for t in batch],
                               dtype=torch.float32, device=self.device)

        # Current Q-values for taken actions
        q_values = self.q_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: select action with online net, evaluate with target net
        with torch.no_grad():
            next_actions = self.q_net(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            targets = rew_t + self.gamma * next_q * (1.0 - done_t)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step_count += 1

        # Periodic target network update
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
        }

    def state_dict(self) -> dict:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state.get("step_count", 0)
