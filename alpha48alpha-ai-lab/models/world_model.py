"""
world_model.py — Alpha48Alpha AI Lab
======================================
A neural forward model that learns to predict the next observation and
reward given a current observation and action.

What is a World Model?
----------------------
A world model is a neural network that internalises environment dynamics:

    (obs_t, action_t)  →  (obs_{t+1},  reward_t)

Once trained on real transitions, the model can simulate imagined roll-outs
entirely inside the network — the foundation of model-based RL methods like
Dreamer, RSSM, and MuZero.

This module provides a compact, self-contained implementation designed to be
used alongside the REINFORCE agent in ``agents/rl_agent.py``.  It is
intentionally kept simple so it is easy to extend with recurrence (GRU/LSTM),
latent state spaces, or probabilistic outputs.

Architecture
------------
    Encoder : [obs  ‖  action_one_hot]  →  hidden  →  hidden
    Head 1  : hidden  →  obs_delta   (predicted change in observation)
    Head 2  : hidden  →  reward_pred (scalar reward prediction)

The model predicts the *delta* (change) rather than the absolute next
observation, which makes learning easier when observations are normalised.

Usage
-----
    from models.world_model import WorldModel

    wm = WorldModel(obs_dim=2, n_actions=4)

    # Collect a transition from the environment, then:
    wm.train_step(obs, action, next_obs, reward)

    # Later, simulate what would happen:
    pred_next_obs, pred_reward = wm.predict(obs, action)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class WorldModel(nn.Module):
    """
    Neural forward model: (obs, action) → (next_obs, reward).

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector (e.g. 2 for SimpleWorld).
    n_actions : int
        Number of discrete actions (used to build a one-hot encoding).
    hidden_dim : int
        Width of both hidden layers.
    lr : float
        Adam learning rate for world-model parameter updates.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Input to the encoder is the concatenation of observation and
        # one-hot encoded action  →  obs_dim + n_actions features
        input_dim = obs_dim + n_actions

        # Shared encoder — extracts a latent representation of (obs, action)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Observation-delta head — predicts next_obs - obs
        self.obs_head = nn.Linear(hidden_dim, obs_dim)

        # Reward head — predicts scalar reward as a single linear output
        self.reward_head = nn.Linear(hidden_dim, 1)

        # Single optimiser for all model parameters
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the world model forward.

        Parameters
        ----------
        obs : torch.Tensor  shape (batch, obs_dim)
            Current observation.
        action_onehot : torch.Tensor  shape (batch, n_actions)
            One-hot encoded action.

        Returns
        -------
        obs_delta : torch.Tensor  shape (batch, obs_dim)
            Predicted change in observation (add to *obs* to get next_obs).
        reward_pred : torch.Tensor  shape (batch, 1)
            Predicted scalar reward.
        """
        # Concatenate observation and action along the feature dimension
        x = torch.cat([obs, action_onehot], dim=-1)
        hidden = self.encoder(x)

        obs_delta = self.obs_head(hidden)
        reward_pred = self.reward_head(hidden)

        return obs_delta, reward_pred

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def train_step(
        self,
        obs: list[float],
        action: int,
        next_obs: list[float],
        reward: float,
    ) -> float:
        """
        Perform one supervised update on a single real transition.

        The model minimises the mean-squared-error on both the
        next-observation prediction and the reward prediction.

        Parameters
        ----------
        obs : list[float]
            Observation before the action.
        action : int
            Discrete action taken.
        next_obs : list[float]
            Observation received after the action.
        reward : float
            Reward received after the action.

        Returns
        -------
        float
            Combined MSE loss (obs loss + reward loss) for logging.
        """
        # Convert inputs to tensors with a batch dimension of 1
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        reward_t = torch.tensor([[reward]], dtype=torch.float32)

        # One-hot encode the action
        action_onehot = self._one_hot(action)

        # Forward pass
        obs_delta_pred, reward_pred = self(obs_t, action_onehot)

        # Target delta = actual next_obs - obs
        obs_delta_target = next_obs_t - obs_t

        # Compute losses
        obs_loss = F.mse_loss(obs_delta_pred, obs_delta_target)
        reward_loss = F.mse_loss(reward_pred, reward_t)
        total_loss = obs_loss + reward_loss

        # Gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    def predict(
        self, obs: list[float], action: int
    ) -> tuple[list[float], float]:
        """
        Predict the next observation and reward for a given (obs, action) pair.

        This can be used to generate imagined roll-outs without interacting
        with the real environment.

        Parameters
        ----------
        obs : list[float]
            Current normalised observation.
        action : int
            Discrete action to simulate.

        Returns
        -------
        next_obs : list[float]
            Predicted next observation (clamped to [0, 1] after delta is applied).
        reward : float
            Predicted scalar reward.
        """
        self.eval()  # switch to inference mode (disables dropout etc.)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_onehot = self._one_hot(action)

            obs_delta_pred, reward_pred = self(obs_t, action_onehot)

            # Apply predicted delta and clamp to valid observation range
            next_obs_t = (obs_t + obs_delta_pred).clamp(0.0, 1.0)

        # Return Python lists/floats for easy use in training loops
        next_obs = next_obs_t.squeeze(0).tolist()
        reward = reward_pred.item()
        return next_obs, reward

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save world-model weights to *path*.

        Parameters
        ----------
        path : str
            File path (e.g. ``"checkpoints/world_model.pt"``).
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load world-model weights from *path*.

        Parameters
        ----------
        path : str
            Path to a previously saved checkpoint.
        """
        self.load_state_dict(torch.load(path, weights_only=True))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _one_hot(self, action: int) -> torch.Tensor:
        """
        Return a (1, n_actions) one-hot tensor for the given action index.

        Parameters
        ----------
        action : int
            Discrete action index.

        Returns
        -------
        torch.Tensor  shape (1, n_actions)
        """
        onehot = torch.zeros(1, self.n_actions, dtype=torch.float32)
        onehot[0, action] = 1.0
        return onehot
