"""ContinuousWorld environment.

A 2-D continuous navigation task where the agent controls its own
velocity to reach a goal region while staying within bounds.

Observation (5-D vector)
------------------------
  [agent_x, agent_y, goal_x, goal_y, distance_to_goal]  (all normalised 0–1)

Actions (2-D continuous box)
-----------------------------
  [delta_x, delta_y]  clipped to [-1, 1]; scaled by ``speed``

Reward
------
  - Dense shaping: negative L2 distance to goal (encourages approach)
  - Goal bonus:    +1.0 on reaching the goal region (radius ``goal_radius``)
  - Step penalty:  small negative constant
  - Out-of-bounds: large negative penalty (rare; agent is clipped to arena)

Extension hooks
---------------
- Add ``obstacles`` (list of circular forbidden zones)
- Override ``_reward()`` for custom shaping (potential-based, curiosity …)
- Add wind / stochastic dynamics for robustness research
- Subclass for multimodal control: attach an image channel via ``render_rgb()``
"""

from __future__ import annotations

import numpy as np

from rl_lab.envs.base import BaseEnv


class ContinuousWorld(BaseEnv):
    """2-D continuous navigation task.

    Parameters
    ----------
    max_steps : int
        Episode truncation limit.
    speed : float
        Maximum displacement per step (fraction of arena width).
    goal_radius : float
        Success threshold distance (fraction of arena width).
    goal_reward : float
        Bonus reward on reaching the goal.
    step_penalty : float
        Per-step penalty.
    """

    def __init__(
        self,
        max_steps: int = 200,
        speed: float = 0.05,
        goal_radius: float = 0.05,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
    ) -> None:
        self.max_steps = max_steps
        self.speed = speed
        self.goal_radius = goal_radius
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

        # Observation: [ax, ay, gx, gy, dist]
        self.observation_space = {"shape": (5,), "dtype": np.float32}
        # Action: [dx, dy] in [-1, 1]
        self.action_space = {
            "low": np.array([-1.0, -1.0], dtype=np.float32),
            "high": np.array([1.0, 1.0], dtype=np.float32),
        }

        self._rng: np.random.Generator = np.random.default_rng()
        self._agent: np.ndarray = np.zeros(2, dtype=np.float32)
        self._goal: np.ndarray = np.ones(2, dtype=np.float32)
        self._prev_dist: float = 0.0
        self._steps: int = 0

    # ------------------------------------------------------------------ #
    #  BaseEnv API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._rng = np.random.default_rng(seed)
        self._steps = 0

        # Place agent and goal at random non-overlapping positions
        while True:
            self._agent = self._rng.uniform(0.0, 1.0, size=2).astype(np.float32)
            self._goal = self._rng.uniform(0.0, 1.0, size=2).astype(np.float32)
            if np.linalg.norm(self._agent - self._goal) > 2 * self.goal_radius:
                break

        self._prev_dist = float(np.linalg.norm(self._agent - self._goal))
        return self._build_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # Normalise diagonal movement so speed is constant
        norm = np.linalg.norm(action)
        if norm > 1e-8:
            action = action / norm * min(norm, 1.0)

        self._agent = np.clip(self._agent + action * self.speed, 0.0, 1.0)
        self._steps += 1

        dist = float(np.linalg.norm(self._agent - self._goal))
        reward = self._reward(dist)
        self._prev_dist = dist

        terminated = dist <= self.goal_radius
        truncated = self._steps >= self.max_steps

        if terminated:
            reward += self.goal_reward

        info = {
            "steps": self._steps,
            "agent_pos": self._agent.tolist(),
            "goal_pos": self._goal.tolist(),
            "dist_to_goal": dist,
        }
        return self._build_obs(), reward, terminated, truncated, info

    def render(self) -> str:
        cols, rows = 20, 10
        grid = [["." for _ in range(cols)] for _ in range(rows)]

        def _place(pos: np.ndarray, symbol: str) -> None:
            c = int(np.clip(pos[0] * (cols - 1), 0, cols - 1))
            r = int(np.clip((1 - pos[1]) * (rows - 1), 0, rows - 1))
            grid[r][c] = symbol

        _place(self._goal, "G")
        _place(self._agent, "A")

        border = "+" + "-" * cols + "+"
        body = "\n".join("|" + "".join(row) + "|" for row in grid)
        dist = np.linalg.norm(self._agent - self._goal)
        footer = f"Step {self._steps}/{self.max_steps}  dist={dist:.3f}"
        return border + "\n" + body + "\n" + border + "\n" + footer

    # ------------------------------------------------------------------ #
    #  Hooks                                                               #
    # ------------------------------------------------------------------ #

    def _reward(self, dist: float) -> float:
        """Dense shaping reward: progress toward goal + per-step penalty."""
        progress = self._prev_dist - dist
        return progress + self.step_penalty

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_obs(self) -> np.ndarray:
        dist = float(np.linalg.norm(self._agent - self._goal))
        return np.array(
            [self._agent[0], self._agent[1], self._goal[0], self._goal[1], dist],
            dtype=np.float32,
        )
