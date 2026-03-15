"""GridWorld environment.

A discrete, fully-observable grid world where an agent must reach a
goal while avoiding optional obstacles and collecting optional rewards.

Features
--------
- Configurable grid size, obstacle density, and episode time-limit
- Multi-channel observation: agent position, goal position, obstacle map,
  step normalisation — giving agents rich state information out-of-the-box
- ASCII rendering for quick debugging and human-in-the-loop research
- Reproducible episode generation via ``seed``

Grid symbols (render)
---------------------
  S  — starting position
  G  — goal
  A  — agent (current position)
  #  — obstacle
  .  — empty cell

Extension hooks
---------------
- Add ``coins`` for sparse multi-objective rewards
- Override ``_reward()`` to shape rewards for curriculum learning
- Add a second agent for competitive / co-operative multi-agent research
"""

from __future__ import annotations

import numpy as np
from typing import Any

from rl_lab.envs.base import BaseEnv


# Cardinal and diagonal moves
_MOVES = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


class GridWorld(BaseEnv):
    """N×N discrete grid navigation task.

    Parameters
    ----------
    size : int
        Grid side length (default 8).
    obstacle_density : float
        Fraction of non-start / non-goal cells that are obstacles (0–0.4).
    max_steps : int
        Episode truncation limit.
    goal_reward : float
        Reward for reaching the goal.
    step_penalty : float
        Penalty applied every step (encourages efficiency).
    obstacle_penalty : float
        Penalty for bumping into a wall / obstacle.
    """

    ACTIONS = 4  # up / down / left / right

    def __init__(
        self,
        size: int = 8,
        obstacle_density: float = 0.15,
        max_steps: int = 200,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
        obstacle_penalty: float = -0.05,
    ) -> None:
        self.size = size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty

        # Observation: 4 channels × size × size  (agent, goal, obstacles, steps)
        # Flattened to a 1-D vector for MLP agents.
        obs_dim = 4 * size * size
        self.observation_space = {
            "shape": (obs_dim,),
            "dtype": np.float32,
        }
        self.action_space = {"n": self.ACTIONS}

        # State (initialised by reset)
        self._rng: np.random.Generator = np.random.default_rng()
        self._grid: np.ndarray = np.zeros((size, size), dtype=np.int8)
        self._agent: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (size - 1, size - 1)
        self._start: tuple[int, int] = (0, 0)
        self._steps: int = 0

    # ------------------------------------------------------------------ #
    #  BaseEnv API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._build_grid()
        return self._build_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if action not in _MOVES:
            raise ValueError(f"Invalid action {action!r}; expected 0–{self.ACTIONS - 1}")

        dr, dc = _MOVES[action]
        nr, nc = self._agent[0] + dr, self._agent[1] + dc

        reward = self.step_penalty
        hit_obstacle = False

        if 0 <= nr < self.size and 0 <= nc < self.size and self._grid[nr, nc] == 0:
            self._agent = (nr, nc)
        else:
            # Bumped into wall or obstacle — stay in place, apply penalty
            hit_obstacle = True
            reward += self.obstacle_penalty

        self._steps += 1
        terminated = self._agent == self._goal
        truncated = self._steps >= self.max_steps

        if terminated:
            reward += self.goal_reward

        info = {
            "steps": self._steps,
            "agent_pos": self._agent,
            "hit_obstacle": hit_obstacle,
        }
        return self._build_obs(), reward, terminated, truncated, info

    def render(self) -> str:
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self._agent:
                    cell = "A"
                elif (r, c) == self._goal:
                    cell = "G"
                elif (r, c) == self._start:
                    cell = "S"
                elif self._grid[r, c] == 1:
                    cell = "#"
                else:
                    cell = "."
                row.append(cell)
            rows.append(" ".join(row))
        header = f"Step {self._steps}/{self.max_steps}"
        return header + "\n" + "\n".join(rows)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_grid(self) -> None:
        """Place start, goal, and random obstacles."""
        self._grid[:] = 0
        self._start = (0, 0)
        self._goal = (self.size - 1, self.size - 1)
        self._agent = self._start

        n_obstacles = int(self.obstacle_density * (self.size ** 2 - 2))
        forbidden = {self._start, self._goal}
        placed = 0
        attempts = 0
        while placed < n_obstacles and attempts < 10 * n_obstacles:
            r = int(self._rng.integers(0, self.size))
            c = int(self._rng.integers(0, self.size))
            if (r, c) not in forbidden:
                self._grid[r, c] = 1
                forbidden.add((r, c))
                placed += 1
            attempts += 1

    def _build_obs(self) -> np.ndarray:
        """Return a flat 4-channel observation vector."""
        size = self.size
        agent_ch = np.zeros((size, size), dtype=np.float32)
        goal_ch = np.zeros((size, size), dtype=np.float32)
        obstacle_ch = self._grid.astype(np.float32)
        step_ch = np.full((size, size), self._steps / self.max_steps, dtype=np.float32)

        agent_ch[self._agent] = 1.0
        goal_ch[self._goal] = 1.0

        return np.concatenate([
            agent_ch.ravel(),
            goal_ch.ravel(),
            obstacle_ch.ravel(),
            step_ch.ravel(),
        ])
