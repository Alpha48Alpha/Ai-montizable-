"""
models/value_network.py — Critic / Q-value network.

Used by both:
  * DQN agents    — outputs one Q-value per discrete action (Q-network).
  * Actor-critic  — outputs a single scalar state-value V(s) (set
                    action_size=1 for this mode).

Architecture: fully-connected MLP with configurable hidden size.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Fully-connected value / Q-network for discrete action spaces.

    Parameters
    ----------
    state_size : int
        Dimensionality of the flattened observation vector.
    action_size : int
        * For DQN  — number of discrete actions (outputs Q(s, a) for each a).
        * For V(s) — set to 1.
    hidden_size : int
        Width of each hidden layer (default: 128).
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values (or state value).

        Parameters
        ----------
        state : torch.Tensor, shape (batch, state_size) or (state_size,)

        Returns
        -------
        torch.Tensor, shape (batch, action_size) or (action_size,)
            Raw Q-values — no activation applied.
        """
        return self.network(state)

    # ------------------------------------------------------------------
    # Action selection helpers (DQN mode)
    # ------------------------------------------------------------------

    def greedy_action(self, state: torch.Tensor) -> int:
        """
        Return the action with the highest Q-value (argmax policy).

        Parameters
        ----------
        state : torch.Tensor, shape (state_size,)

        Returns
        -------
        int — index of the greedy action.
        """
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))   # (1, action_size)
        return q_values.argmax(dim=-1).item()
