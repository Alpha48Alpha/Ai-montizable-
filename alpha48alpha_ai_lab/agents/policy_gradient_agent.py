"""
policy_gradient_agent.py — REINFORCE policy-gradient agent.

Uses the NeuralPolicy model and updates parameters by maximising the expected
cumulative discounted reward (the REINFORCE algorithm with a baseline).
"""

from typing import List, Tuple

import torch
import torch.optim as optim

from alpha48alpha_ai_lab.config import GAMMA, LEARNING_RATE
from alpha48alpha_ai_lab.models.neural_policy import NeuralPolicy


class PolicyGradientAgent:
    """REINFORCE agent that learns a stochastic policy via policy gradient.

    The agent collects full episode trajectories, computes discounted returns,
    and updates the policy network to increase the log-probability of actions
    that led to high returns.
    """

    def __init__(
        self,
        state_size: int,
        num_actions: int,
        hidden_size: int = 128,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
    ) -> None:
        """
        Initialize the policy-gradient agent.

        Args:
            state_size:    Total number of states (for one-hot encoding).
            num_actions:   Number of discrete actions.
            hidden_size:   Hidden layer size for the policy network.
            learning_rate: Adam optimiser learning rate.
            gamma:         Discount factor for future rewards.
        """
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma

        self.policy = NeuralPolicy(state_size, num_actions, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Buffers filled during an episode, cleared after each update
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Sample an action from the current policy.

        Also stores the log-probability for the subsequent update step.

        Args:
            state: Integer index of the current state.

        Returns:
            Sampled action index.
        """
        action, log_prob = self.policy.select_action(state, self.state_size)
        self._log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float) -> None:
        """Record the reward for the most recent step.

        Args:
            reward: Scalar reward from the environment.
        """
        self._rewards.append(reward)

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self) -> float:
        """Perform a single REINFORCE policy-gradient update.

        Computes discounted returns, normalises them, and back-propagates
        the policy loss.

        Returns:
            Scalar policy loss value (for logging).
        """
        returns = self._compute_returns()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack(
            [-log_p * ret for log_p, ret in zip(self._log_probs, returns)]
        ).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode buffers
        self._log_probs.clear()
        self._rewards.clear()

        return loss.item()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_returns(self) -> torch.Tensor:
        """Compute discounted cumulative returns for the stored episode.

        Returns:
            1-D tensor of discounted returns, one per time step.
        """
        returns: List[float] = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)
