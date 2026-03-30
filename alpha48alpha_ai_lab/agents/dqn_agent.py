"""
dqn_agent.py — Deep Q-Network (DQN) agent.

Implements the classic DQN algorithm (Mnih et al., 2015) with:
    - experience replay (via ReplayBuffer)
    - a separate target network for stable Q-learning targets
    - ε-greedy exploration with exponential decay
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

from alpha48alpha_ai_lab.config import (
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    LEARNING_RATE,
    REPLAY_BUFFER_SIZE,
    TARGET_UPDATE_FREQ,
)
from alpha48alpha_ai_lab.models.value_network import QNetwork
from alpha48alpha_ai_lab.training.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent using a Q-network and experience replay.

    Call ``select_action`` at every step, ``store_transition`` after each
    environment step, and ``update`` after accumulating enough transitions.
    Call ``sync_target`` every ``TARGET_UPDATE_FREQ`` episodes.
    """

    def __init__(
        self,
        state_size: int,
        num_actions: int,
        hidden_size: int = 128,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        batch_size: int = BATCH_SIZE,
        buffer_size: int = REPLAY_BUFFER_SIZE,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        epsilon_decay: float = EPSILON_DECAY,
    ) -> None:
        """
        Initialize the DQN agent.

        Args:
            state_size:     Total number of states (for one-hot encoding).
            num_actions:    Number of discrete actions.
            hidden_size:    Hidden layer size for the Q-networks.
            learning_rate:  Adam optimiser learning rate.
            gamma:          Discount factor for future rewards.
            batch_size:     Mini-batch size for each gradient update.
            buffer_size:    Maximum size of the replay buffer.
            epsilon_start:  Initial exploration probability.
            epsilon_end:    Minimum exploration probability.
            epsilon_decay:  Per-episode multiplicative decay for ε.
        """
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Online network (trained at every update step)
        self.q_network = QNetwork(state_size, num_actions, hidden_size)
        # Target network (periodically synced with the online network)
        self.target_network = QNetwork(state_size, num_actions, hidden_size)
        self.sync_target()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Choose an action using ε-greedy exploration.

        Args:
            state: Integer index of the current state.

        Returns:
            Selected action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        one_hot = torch.zeros(1, self.state_size)
        one_hot[0, state] = 1.0
        with torch.no_grad():
            q_values = self.q_network(one_hot)
        return int(q_values.argmax(dim=1).item())

    def decay_epsilon(self) -> None:
        """Decay ε after each episode, clamped at ``epsilon_end``."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Replay & learning
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer.

        Args:
            state:      Current state index.
            action:     Action taken.
            reward:     Reward received.
            next_state: Next state index.
            done:       Whether the episode ended.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> float:
        """Sample a mini-batch and perform one DQN gradient update.

        Does nothing if the replay buffer has fewer transitions than the
        batch size.

        Returns:
            Scalar TD loss (or 0.0 if the update was skipped).
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.state_size
        )

        # Current Q-values for the taken actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target(self) -> None:
        """Copy weights from the online network to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
