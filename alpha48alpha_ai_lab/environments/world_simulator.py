"""
environments/world_simulator.py — 2-D world simulator with obstacles & goals.

The WorldSimulator extends the basic grid-world concept with:
  * Randomly or explicitly placed wall / obstacle cells.
  * Multiple possible goal locations.
  * Richer state representation (one-hot encoded cell type in a local
    3×3 neighbourhood around the agent).

Action space (same as GridWorld)
---------------------------------
  0 — UP    1 — DOWN    2 — LEFT    3 — RIGHT

State representation
--------------------
The state vector encodes:
  1. Normalised agent position: [row/(H-1), col/(W-1)]          — 2 values
  2. Normalised Manhattan distance to nearest goal               — 1 value
  3. Local 3×3 occupancy map (obstacle=1, free=0, goal=2/9)      — 9 values
  ─────────────────────────────────────────────────────────────────────────
  Total: 12 float values

Cell type encoding (internal)
------------------------------
  0 — free space
  1 — obstacle / wall
  2 — goal
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np


# Action index → (delta_row, delta_col)
_ACTION_DELTAS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}

# Cell type constants
_FREE     = 0
_OBSTACLE = 1
_GOAL     = 2


class WorldSimulator:
    """
    2-D world simulator with walls, obstacles, and goal states.

    Parameters
    ----------
    width : int
        Number of columns in the world grid.
    height : int
        Number of rows in the world grid.
    num_obstacles : int
        Number of randomly placed obstacle cells (0 disables obstacles).
    num_goals : int
        Number of randomly placed goal cells (minimum 1).
    reward_goal : float
        Reward awarded when the agent steps onto a goal cell.
    reward_step : float
        Per-step reward (typically a small negative value).
    reward_obstacle : float
        Reward (penalty) for attempting to move into an obstacle.
    max_steps : int
        Episode length limit.
    seed : Optional[int]
        Random seed for reproducible obstacle / goal placement.
    """

    def __init__(
        self,
        width:           int   = 10,
        height:          int   = 10,
        num_obstacles:   int   = 10,
        num_goals:       int   = 1,
        reward_goal:     float = 10.0,
        reward_step:     float = -0.1,
        reward_obstacle: float = -1.0,
        max_steps:       int   = 200,
        seed:            Optional[int] = None,
    ):
        self.width           = width
        self.height          = height
        self.num_obstacles   = num_obstacles
        self.num_goals       = num_goals
        self.reward_goal     = reward_goal
        self.reward_step     = reward_step
        self.reward_obstacle = reward_obstacle
        self.max_steps       = max_steps
        self._seed           = seed

        # Will be populated by reset()
        self._grid: np.ndarray           = np.zeros((height, width), dtype=np.int8)
        self._agent_row: int             = 0
        self._agent_col: int             = 0
        self._goal_positions: List[Tuple[int, int]] = []
        self._step_count: int            = 0
        self._rng                        = random.Random(seed)

        # Build the initial world layout
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the world: regenerate obstacles & goals, place the agent.

        Returns
        -------
        np.ndarray, shape (12,)
            Current state observation.
        """
        self._rng = random.Random(self._seed)   # reset RNG for reproducibility
        self._step_count = 0
        self._grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Collect all (row, col) positions; shuffle for random placement
        all_cells = [
            (r, c) for r in range(self.height) for c in range(self.width)
        ]
        self._rng.shuffle(all_cells)

        # Place obstacles (skip first cell — reserved for agent start)
        obstacle_cells = all_cells[1 : 1 + self.num_obstacles]
        for r, c in obstacle_cells:
            self._grid[r, c] = _OBSTACLE

        # Place goal(s)
        used = set(obstacle_cells) | {all_cells[0]}
        goal_candidates = [pos for pos in all_cells if pos not in used]
        self._goal_positions = goal_candidates[: self.num_goals]
        for r, c in self._goal_positions:
            self._grid[r, c] = _GOAL

        # Place agent at the first free cell
        self._agent_row, self._agent_col = all_cells[0]

        return self.get_state()

    def step(self, action: int):
        """
        Apply an action and advance the world by one time-step.

        Parameters
        ----------
        action : int  — one of {0, 1, 2, 3}

        Returns
        -------
        next_state : np.ndarray, shape (12,)
        reward : float
        done : bool
        info : dict
        """
        if action not in _ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Must be in {{0,1,2,3}}.")

        dr, dc = _ACTION_DELTAS[action]
        new_row = self._agent_row + dr
        new_col = self._agent_col + dc

        # Check boundary
        if not (0 <= new_row < self.height and 0 <= new_col < self.width):
            # Agent stays; boundary penalty
            reward = self.reward_obstacle
            self._step_count += 1
            done = self._step_count >= self.max_steps
            return self.get_state(), reward, done, {"steps": self._step_count, "event": "boundary"}

        cell_type = self._grid[new_row, new_col]

        if cell_type == _OBSTACLE:
            # Agent stays; obstacle penalty
            reward = self.reward_obstacle
            self._step_count += 1
            done = self._step_count >= self.max_steps
            return self.get_state(), reward, done, {"steps": self._step_count, "event": "obstacle"}

        # Move the agent
        self._agent_row = new_row
        self._agent_col = new_col
        self._step_count += 1

        if cell_type == _GOAL:
            reward = self.reward_goal
            done   = True
            return self.get_state(), reward, done, {"steps": self._step_count, "event": "goal"}

        # Normal free-space step
        reward = self.reward_step
        done   = self._step_count >= self.max_steps
        return self.get_state(), reward, done, {"steps": self._step_count, "event": "step"}

    def render(self) -> None:
        """Print an ASCII representation of the world."""
        symbols = {_FREE: ".", _OBSTACLE: "#", _GOAL: "G"}
        print(f"\n--- WorldSimulator {self.height}×{self.width} | Step {self._step_count} ---")
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                if r == self._agent_row and c == self._agent_col:
                    row_str += " A "
                else:
                    row_str += f" {symbols[int(self._grid[r, c])]} "
            print(row_str)
        print()

    def get_state(self) -> np.ndarray:
        """
        Construct the observation vector for the current world state.

        Returns
        -------
        np.ndarray, shape (12,)
        """
        # 1. Normalised agent position
        row_norm = self._agent_row / max(self.height - 1, 1)
        col_norm = self._agent_col / max(self.width  - 1, 1)

        # 2. Normalised Manhattan distance to nearest goal
        if self._goal_positions:
            dist = min(
                abs(self._agent_row - gr) + abs(self._agent_col - gc)
                for gr, gc in self._goal_positions
            )
            max_dist = (self.height - 1) + (self.width - 1)
            dist_norm = dist / max_dist if max_dist > 0 else 0.0
        else:
            dist_norm = 1.0

        # 3. Local 3×3 occupancy map (obstacle=1/9, goal=2/9, free=0)
        neighbourhood = np.zeros(9, dtype=np.float32)
        idx = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = self._agent_row + dr
                nc = self._agent_col + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    neighbourhood[idx] = float(self._grid[nr, nc]) / 9.0
                else:
                    neighbourhood[idx] = float(_OBSTACLE) / 9.0  # out-of-bounds as wall
                idx += 1

        state = np.array([row_norm, col_norm, dist_norm], dtype=np.float32)
        return np.concatenate([state, neighbourhood])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """Dimensionality of the observation vector."""
        return 12

    @property
    def action_size(self) -> int:
        """Number of discrete actions."""
        return 4
