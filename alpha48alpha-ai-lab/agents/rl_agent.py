"""
rl_agent.py — Alpha48Alpha AI Lab
===================================
Reinforcement learning agent using the REINFORCE (vanilla policy gradient)
algorithm with PyTorch.

How it works
------------
1.  The agent collects a full episode of (state, action, reward) transitions.
2.  Discounted returns G_t are computed backwards through the trajectory.
3.  The policy network parameters θ are updated by ascending the gradient of:

        J(θ) = E[ Σ_t  log π_θ(a_t | s_t) · G_t ]

    which is equivalent to minimising  -J(θ).

4.  Optional baseline subtraction (mean return) reduces variance.

References
----------
Williams, R.J. (1992). Simple Statistical Gradient-Following Algorithms for
Connectionist Reinforcement Learning. Machine Learning, 8, 229–256.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    A small fully-connected neural network that maps observations to a
    probability distribution over actions.

    Architecture:  input → Linear → ReLU → Linear → Softmax

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector.
    n_actions : int
        Number of discrete actions.
    hidden_dim : int
        Width of the single hidden layer.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, obs_dim)
            Observation(s).

        Returns
        -------
        torch.Tensor  shape (batch, n_actions)
            Action probability distribution (after softmax).
        """
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class RLAgent:
    """
    REINFORCE agent that wraps a PolicyNetwork and implements the
    policy-gradient update rule.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector.
    n_actions : int
        Number of discrete actions.
    hidden_dim : int
        Width of the hidden layer in the policy network.
    lr : float
        Adam optimiser learning rate.
    gamma : float
        Discount factor for future rewards (0 < γ ≤ 1).
    use_baseline : bool
        If True, subtract the mean return from each G_t to reduce variance.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True,
    ) -> None:
        self.gamma = gamma
        self.use_baseline = use_baseline

        # Policy network and optimiser
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode memory — cleared after each update
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    # ------------------------------------------------------------------
    # Interaction methods (called every step)
    # ------------------------------------------------------------------

    def select_action(self, observation: list[float]) -> int:
        """
        Sample an action from the current policy.

        The log-probability of the selected action is stored internally
        and used later in ``update()``.

        Parameters
        ----------
        observation : list[float]
            Raw observation from the environment.

        Returns
        -------
        int
            Sampled action index.
        """
        # Convert observation to a float tensor with a batch dimension
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Get action probabilities from the policy network
        probs = self.policy(obs_tensor)

        # Create a categorical distribution and sample
        dist = Categorical(probs)
        action = dist.sample()

        # Store log-probability for the policy-gradient update
        self._log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward: float) -> None:
        """
        Record the reward received after the most recent action.

        Parameters
        ----------
        reward : float
            Scalar reward from the environment.
        """
        self._rewards.append(reward)

    # ------------------------------------------------------------------
    # Learning (called at the end of each episode)
    # ------------------------------------------------------------------

    def update(self) -> float:
        """
        Perform one policy-gradient update using the stored episode data.

        Steps
        -----
        1. Compute discounted returns G_t for every time-step.
        2. Optionally subtract the mean return (baseline) to reduce variance.
        3. Compute the policy-gradient loss:  -Σ_t  log π(a_t|s_t) · G_t
        4. Back-propagate and update the policy network weights.
        5. Clear episode memory.

        Returns
        -------
        float
            Scalar policy-gradient loss for logging purposes.
        """
        # ---- 1. Compute discounted returns --------------------------------
        returns: list[float] = []
        G = 0.0
        # Traverse rewards in reverse to accumulate discounted sum
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # Convert to tensor for vectorised operations
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # ---- 2. Variance reduction via mean baseline ----------------------
        if self.use_baseline and len(returns_tensor) > 1:
            returns_tensor = returns_tensor - returns_tensor.mean()

        # ---- 3. Policy-gradient loss  -Σ log π(a|s) · G_t ---------------
        log_probs_tensor = torch.stack(self._log_probs)
        loss = -(log_probs_tensor * returns_tensor).sum()

        # ---- 4. Gradient update -------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ---- 5. Clear episode memory for the next episode -----------------
        self._log_probs = []
        self._rewards = []

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the policy network weights to *path*.

        Parameters
        ----------
        path : str
            File path (e.g. ``"checkpoints/policy.pt"``).
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load policy network weights from *path*.

        Parameters
        ----------
        path : str
            File path to a previously saved checkpoint.
        """
        self.policy.load_state_dict(torch.load(path, weights_only=True))
        self.policy.eval()
