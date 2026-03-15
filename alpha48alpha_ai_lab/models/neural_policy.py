"""
models/neural_policy.py — Stochastic actor policy network.

The NeuralPolicy takes a state observation as input and outputs a
probability distribution over discrete actions.  During training,
actions are sampled from this distribution; at evaluation time the
greedy (argmax) action can be used.

Architecture: fully-connected MLP with configurable hidden size.
Output layer applies a softmax activation so the output is a valid
probability vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class NeuralPolicy(nn.Module):
    """
    Fully-connected policy network for discrete action spaces.

    Parameters
    ----------
    state_size : int
        Dimensionality of the flattened observation vector.
    action_size : int
        Number of discrete actions available to the agent.
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
        Compute action-probability vector.

        Parameters
        ----------
        state : torch.Tensor, shape (batch, state_size) or (state_size,)

        Returns
        -------
        torch.Tensor, shape (batch, action_size) or (action_size,)
            Probability distribution over actions (softmax applied).
        """
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Action selection helpers
    # ------------------------------------------------------------------

    def select_action(self, state: torch.Tensor):
        """
        Sample an action from the policy distribution.

        Parameters
        ----------
        state : torch.Tensor, shape (state_size,)
            Single (unbatched) observation.

        Returns
        -------
        action : int
            Sampled action index.
        log_prob : torch.Tensor
            Log-probability of the sampled action (used for policy gradient).
        """
        probs = self.forward(state.unsqueeze(0))          # (1, action_size)
        dist   = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def greedy_action(self, state: torch.Tensor) -> int:
        """
        Select the highest-probability action (no exploration).

        Parameters
        ----------
        state : torch.Tensor, shape (state_size,)

        Returns
        -------
        int — index of the greedy action.
        """
        with torch.no_grad():
            probs = self.forward(state.unsqueeze(0))
        return probs.argmax(dim=-1).item()
