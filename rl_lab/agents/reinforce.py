"""REINFORCE (Monte-Carlo policy-gradient) agent.

Implements the Williams (1992) REINFORCE algorithm with:
  - Learned baseline (value network) to reduce variance
  - Entropy bonus to encourage exploration
  - Gradient clipping for training stability
  - Configurable discount (gamma) and learning rates

Supports both **discrete** (GridWorld) and **continuous Gaussian**
(ContinuousWorld) action spaces in a single class, selected
automatically via ``is_discrete``.

Algorithm sketch (discrete)
---------------------------
  for episode in range(N):
      collect trajectory τ = (s₀, a₀, r₁, s₁, …, sT) on-policy
      compute discounted returns G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}
      baseline b_t  = V(s_t)  (critic prediction)
      advantage A_t = G_t - b_t
      loss = -Σ_t log π(a_t|s_t) · A_t  (actor)
           + 0.5 · Σ_t (G_t - b_t)²     (critic)
           - β · H[π(·|s_t)]             (entropy bonus)
      update θ via Adam

Extension hooks
---------------
- Swap ``PolicyMLP`` for ``CNNEncoder + head`` to handle image observations
- Replace the trajectory with a rollout buffer for PPO-style updates
- Attach a world model to generate imaginary rollouts
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_lab.agents.base import BaseAgent
from rl_lab.models.mlp import PolicyMLP, ValueMLP


class REINFORCEAgent(BaseAgent):
    """Monte-Carlo policy gradient with value-function baseline.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    is_discrete : bool
        True → categorical policy; False → diagonal Gaussian policy.
    hidden_sizes : list[int]
        Hidden layer widths for both actor and critic.
    actor_lr : float
    critic_lr : float
    gamma : float
        Discount factor.
    entropy_coef : float
        Entropy regularisation coefficient β.
    grad_clip : float | None
        Maximum gradient norm (None = no clipping).
    device : str
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        is_discrete: bool = True,
        hidden_sizes: list[int] | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        grad_clip: float | None = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_dim, act_dim, device)
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.is_discrete = is_discrete
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

        # Actor: outputs logits (discrete) or (mean, log_std) (continuous)
        actor_out = act_dim if is_discrete else act_dim * 2
        self.actor = PolicyMLP(obs_dim, actor_out, hidden_sizes).to(self.device)
        self.critic = ValueMLP(obs_dim, hidden_sizes).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Episode buffer
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[Any] = []
        self._rew_buf: list[float] = []
        self._logp_buf: list[torch.Tensor] = []
        self._entropy_buf: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    #  BaseAgent API                                                       #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: np.ndarray) -> Any:
        """Sample an action from the current policy."""
        obs_t = self.preprocess_obs(obs).unsqueeze(0)  # (1, obs_dim)
        # NOTE: do NOT use torch.no_grad() here — we need grad_fn on logp
        # for the policy-gradient update later in update().
        raw = self.actor(obs_t)

        dist = self._make_dist(raw)
        action_t = dist.sample()
        logp = dist.log_prob(action_t)
        if not self.is_discrete:
            logp = logp.sum(-1)
        entropy = dist.entropy()
        if not self.is_discrete:
            entropy = entropy.sum(-1)

        self._obs_buf.append(obs)
        if self.is_discrete:
            action = int(action_t.item())
        else:
            action = action_t.squeeze(0).numpy()
        self._act_buf.append(action)
        self._logp_buf.append(logp)
        self._entropy_buf.append(entropy)
        return action

    def store_reward(self, reward: float) -> None:
        """Call after each ``env.step()`` to record the reward."""
        self._rew_buf.append(reward)

    def update(self) -> dict:  # type: ignore[override]
        """Run one REINFORCE update over the collected episode trajectory."""
        if not self._rew_buf:
            return {}

        returns = self._compute_returns()
        obs_t = torch.stack([self.preprocess_obs(o) for o in self._obs_buf])
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        logp_t = torch.stack(self._logp_buf).to(self.device)
        ent_t = torch.stack(self._entropy_buf).to(self.device)

        # ---- Critic update ----
        values = self.critic(obs_t).squeeze(-1)
        critic_loss = nn.functional.mse_loss(values, ret_t)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        # ---- Actor update ----
        with torch.no_grad():
            baselines = self.critic(obs_t).squeeze(-1)
        advantages = ret_t - baselines
        # Normalise advantages for stability
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(logp_t * advantages).mean()
        entropy_loss = -self.entropy_coef * ent_t.mean()
        total_actor_loss = actor_loss + entropy_loss

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": ent_t.mean().item(),
            "mean_return": ret_t.mean().item(),
        }
        self._clear_buffers()
        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_dist(self, raw: torch.Tensor):
        if self.is_discrete:
            return torch.distributions.Categorical(logits=raw)
        mean, log_std = raw.chunk(2, dim=-1)
        log_std = log_std.clamp(-4, 2)
        return torch.distributions.Normal(mean, log_std.exp())

    def _compute_returns(self) -> list[float]:
        G = 0.0
        returns: list[float] = []
        for r in reversed(self._rew_buf):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def _clear_buffers(self) -> None:
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()
        self._logp_buf.clear()
        self._entropy_buf.clear()
