"""
neural_policy.py — Neural network policy for policy-gradient agents.

The policy network maps a state observation to a probability distribution
over actions using a simple feed-forward architecture with ReLU activations
and a softmax output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralPolicy(nn.Module):
    """Feed-forward policy network.

    Input:  one-hot encoded state vector of size ``state_size``.
    Output: probability distribution over ``num_actions`` actions.
    """

    def __init__(self, state_size: int, num_actions: int, hidden_size: int = 128) -> None:
        """
        Initialize the policy network.

        Args:
            state_size:   Dimensionality of the (one-hot) state input.
            num_actions:  Number of discrete actions.
            hidden_size:  Number of units in each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities for the given state batch.

        Args:
            state: Float tensor of shape ``(batch, state_size)`` — one-hot encoded.

        Returns:
            Probability tensor of shape ``(batch, num_actions)``.
        """
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

    def select_action(self, state_idx: int, state_size: int) -> tuple:
        """Sample an action from the policy given a scalar state index.

        Args:
            state_idx:  Integer index of the current state.
            state_size: Total number of states (for one-hot encoding).

        Returns:
            Tuple of (action: int, log_prob: torch.Tensor).
        """
        one_hot = torch.zeros(1, state_size)
        one_hot[0, state_idx] = 1.0
        probs = self.forward(one_hot)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
