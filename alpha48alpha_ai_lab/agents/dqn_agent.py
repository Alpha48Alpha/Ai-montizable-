"""
agents/dqn_agent.py — Deep Q-Network (DQN) agent.

Implements the DQN algorithm (Mnih et al., 2015) with:
  * Experience replay buffer                — breaks temporal correlations
  * Separate target network                 — stabilises training targets
  * ε-greedy exploration schedule           — balances explore / exploit

The agent maintains two networks:
  * online_network  — updated every training step (via replay sampling).
  * target_network  — periodically synced from the online network.

Reference: Mnih et al. (2015). Human-level control through deep
           reinforcement learning. Nature, 518(7540), 529–533.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.value_network import ValueNetwork
from training.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and a target network.

    Parameters
    ----------
    state_size : int
        Dimensionality of the observation vector.
    action_size : int
        Number of discrete actions.
    learning_rate : float
        Adam optimiser learning rate (default: 5e-4).
    gamma : float
        Discount factor (default: 0.99).
    epsilon : float
        Initial exploration rate (default: 1.0).
    epsilon_min : float
        Minimum exploration rate (default: 0.05).
    epsilon_decay : float
        Multiplicative decay applied after each episode (default: 0.995).
    batch_size : int
        Mini-batch size for each training step (default: 64).
    replay_capacity : int
        Maximum number of transitions in the replay buffer (default: 10 000).
    target_update : int
        Number of episodes between target-network synchronisations (default: 10).
    hidden_size : int
        Hidden-layer width of the value network (default: 128).
    """

    def __init__(
        self,
        state_size:      int,
        action_size:     int,
        learning_rate:   float = 5e-4,
        gamma:           float = 0.99,
        epsilon:         float = 1.0,
        epsilon_min:     float = 0.05,
        epsilon_decay:   float = 0.995,
        batch_size:      int   = 64,
        replay_capacity: int   = 10_000,
        target_update:   int   = 10,
        hidden_size:     int   = 128,
    ):
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        # Online and target networks
        self.online_network = ValueNetwork(state_size, action_size, hidden_size)
        self.target_network = copy.deepcopy(self.online_network)
        self.target_network.eval()   # target network is never trained directly

        self.optimizer   = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn     = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Episode counter for target-network sync
        self._episodes_done: int = 0

    # ------------------------------------------------------------------
    # Episode interaction
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Choose an action using an ε-greedy policy.

        With probability ε a random action is chosen (exploration);
        otherwise the action with the highest Q-value is selected (exploitation).

        Parameters
        ----------
        state : np.ndarray, shape (state_size,)

        Returns
        -------
        int — selected action index.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)   # explore

        state_tensor = torch.FloatTensor(state)
        return self.online_network.greedy_action(state_tensor)   # exploit

    def store_transition(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Push a single (s, a, r, s', done) transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Policy update (call once per episode or once per step)
    # ------------------------------------------------------------------

    def update(self) -> float:
        """
        Sample a mini-batch from the replay buffer and perform a DQN update.

        The Bellman target is:
            y = r  +  γ * max_{a'} Q_target(s', a')   (if not done)
            y = r                                       (if done)

        Returns
        -------
        float — scalar TD loss value, or 0.0 if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)
        dones       = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values for the actions that were taken
        current_q = self.online_network(states).gather(1, actions)

        # Target Q-values using the frozen target network
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1, keepdim=True).values
            target_q   = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def end_episode(self) -> None:
        """
        Call at the end of every episode to:
          * Decay ε (reduce exploration).
          * Optionally sync the target network.
        """
        self._episodes_done += 1

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodically copy online → target network
        if self._episodes_done % self.target_update == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the online network weights to ``path``."""
        torch.save(self.online_network.state_dict(), path)

    def load(self, path: str) -> None:
        """Load online (and target) network weights from ``path``."""
        self.online_network.load_state_dict(torch.load(path, weights_only=True))
        self.target_network.load_state_dict(self.online_network.state_dict())
