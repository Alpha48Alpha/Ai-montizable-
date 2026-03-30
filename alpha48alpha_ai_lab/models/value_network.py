"""
value_network.py — Neural network value function for actor-critic and DQN agents.

Two classes are provided:
    - ``ValueNetwork``:  state-value V(s) estimator (scalar output).
    - ``QNetwork``:      action-value Q(s, a) estimator (vector output, one per action).
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """State-value function V(s).

    Input:  one-hot encoded state vector of size ``state_size``.
    Output: scalar value estimate V(s).
    """

    def __init__(self, state_size: int, hidden_size: int = 128) -> None:
        """
        Initialize the value network.

        Args:
            state_size:  Dimensionality of the (one-hot) state input.
            hidden_size: Number of units in each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return the scalar value estimate V(s).

        Args:
            state: Float tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, 1)``.
        """
        return self.net(state)


class QNetwork(nn.Module):
    """Action-value function Q(s, a) for DQN.

    Input:  one-hot encoded state vector of size ``state_size``.
    Output: Q-value for every action — shape ``(batch, num_actions)``.
    """

    def __init__(self, state_size: int, num_actions: int, hidden_size: int = 128) -> None:
        """
        Initialize the Q-network.

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
        """Return Q-values for all actions.

        Args:
            state: Float tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, num_actions)``.
        """
        return self.net(state)
