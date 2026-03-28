"""
agents/policy_gradient_agent.py — REINFORCE policy-gradient agent.

REINFORCE (Williams, 1992) is the classic Monte-Carlo policy gradient
algorithm.  The agent:

  1. Rolls out a full episode, collecting (log_prob, reward) tuples.
  2. Computes discounted returns G_t for each time-step.
  3. Subtracts a baseline (running mean return) to reduce variance.
  4. Updates the policy network by maximising E[G_t * log π(a|s)].

Reference: Williams, R.J. (1992). Simple statistical gradient-following
           algorithms for connectionist reinforcement learning.
           Machine Learning, 8(3-4), 229–256.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim

from models.neural_policy import NeuralPolicy


class PolicyGradientAgent:
    """
    REINFORCE agent with a running-mean baseline.

    Parameters
    ----------
    state_size : int
        Dimensionality of the observation vector.
    action_size : int
        Number of discrete actions.
    learning_rate : float
        Adam optimiser learning rate (default: 1e-3).
    gamma : float
        Discount factor for future rewards (default: 0.99).
    hidden_size : int
        Hidden-layer width of the policy network (default: 128).
    """

    def __init__(
        self,
        state_size:    int,
        action_size:   int,
        learning_rate: float = 1e-3,
        gamma:         float = 0.99,
        hidden_size:   int   = 128,
    ):
        self.gamma = gamma

        self.policy    = NeuralPolicy(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Running baseline (exponential moving average of episode returns)
        self._baseline: float = 0.0
        self._baseline_alpha: float = 0.1   # EMA smoothing coefficient

        # Storage for the current episode
        self._log_probs: List[torch.Tensor] = []
        self._rewards:   List[float]        = []

    # ------------------------------------------------------------------
    # Episode interaction
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Sample an action from the current policy.

        Parameters
        ----------
        state : np.ndarray, shape (state_size,)

        Returns
        -------
        int — selected action index.
        """
        state_tensor = torch.FloatTensor(state)
        action, log_prob = self.policy.select_action(state_tensor)
        self._log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float) -> None:
        """Append a step reward to the current episode buffer."""
        self._rewards.append(reward)

    # ------------------------------------------------------------------
    # Policy update (called once per episode)
    # ------------------------------------------------------------------

    def update(self) -> float:
        """
        Perform a REINFORCE policy-gradient update using the stored episode.

        Returns
        -------
        float — scalar policy loss value (for logging).
        """
        # Compute discounted returns G_t = Σ γ^k * r_{t+k}
        returns = self._compute_returns()

        # Subtract baseline and normalise for stable training
        episode_return = returns[0].item()
        self._baseline = (
            (1 - self._baseline_alpha) * self._baseline
            + self._baseline_alpha * episode_return
        )
        returns = returns - self._baseline

        # Normalise returns (zero mean, unit std) for numerical stability
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss: -E[G_t * log π(a_t | s_t)]
        policy_loss = torch.stack(
            [-lp * ret for lp, ret in zip(self._log_probs, returns)]
        ).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        loss_val = policy_loss.item()

        # Clear episode buffers
        self._log_probs = []
        self._rewards   = []

        return loss_val

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy network weights to ``path``."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load policy network weights from ``path``."""
        self.policy.load_state_dict(torch.load(path, weights_only=True))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_returns(self) -> torch.Tensor:
        """
        Compute discounted returns G_t for all time-steps in the episode.

        Returns
        -------
        torch.Tensor, shape (T,)
        """
        T = len(self._rewards)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G
        return returns
