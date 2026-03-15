"""
Policy Gradient RL Agent
========================
A REINFORCE (vanilla policy gradient) agent implemented with PyTorch.

The agent maintains a simple fully-connected policy network that maps
environment observations to action probabilities. At the end of each
episode the network is updated using the discounted return as the
gradient signal.

References
----------
Williams, R. J. (1992). Simple statistical gradient-following algorithms
for connectionist reinforcement learning. Machine Learning, 8(3–4), 229–256.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Simple two-layer fully-connected policy network."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Parameters
        ----------
        state_size  : Dimension of the input observation.
        action_size : Number of discrete actions.
        hidden_size : Number of units in the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for each action."""
        return self.net(x)


class PolicyGradientAgent:
    """REINFORCE agent that learns a stochastic policy via policy gradients."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        """
        Parameters
        ----------
        state_size  : Dimension of the flat observation vector.
        action_size : Number of discrete actions available in the environment.
        hidden_size : Hidden layer width of the policy network.
        lr          : Learning rate for the Adam optimizer.
        gamma       : Discount factor for computing returns.
        """
        self.gamma = gamma

        # Policy network and optimizer
        self.policy = PolicyNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Per-episode storage: log-probabilities and rewards
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, observation: list[float]) -> int:
        """Sample an action from the policy given an observation.

        The selected action's log-probability is stored internally for
        the upcoming gradient update.

        Parameters
        ----------
        observation : Flat observation vector from the environment.

        Returns
        -------
        int
            Chosen action index.
        """
        state = torch.tensor(observation, dtype=torch.float32)
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float) -> None:
        """Record the reward received after the last action."""
        self._rewards.append(reward)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update_policy(self) -> float:
        """Compute the policy gradient loss and update the network weights.

        Should be called once per episode, after all steps are complete.

        Returns
        -------
        float
            The scalar policy loss value (for logging).
        """
        # Compute discounted returns G_t for each timestep
        returns: list[float] = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # Normalise returns for training stability
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        if returns_tensor.std() > 1e-8:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Policy gradient loss: -E[log π(a|s) * G_t]
        log_probs_tensor = torch.stack(self._log_probs)
        loss = -(log_probs_tensor * returns_tensor).sum()

        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode buffers
        self._log_probs = []
        self._rewards = []

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the policy network weights to a file."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load policy network weights from a file."""
        self.policy.load_state_dict(torch.load(path, weights_only=True))
        self.policy.eval()
