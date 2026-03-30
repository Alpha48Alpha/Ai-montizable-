"""
grid_world.py — 2-D Grid World environment for reinforcement learning.

The agent starts at the top-left corner (0, 0) and must reach the goal
state at the bottom-right corner (GRID_SIZE-1, GRID_SIZE-1) while avoiding
randomly placed obstacles.

Action space (discrete, 4 actions):
    0 = UP    (row - 1)
    1 = DOWN  (row + 1)
    2 = LEFT  (col - 1)
    3 = RIGHT (col + 1)

Observation:
    Flat integer index of the agent's current cell: row * GRID_SIZE + col.
"""

import random
from typing import List, Optional, Tuple

import numpy as np

from alpha48alpha_ai_lab.config import GRID_SIZE, MAX_STEPS


class GridWorld:
    """Simple 2-D grid world compatible with RL training loops."""

    # Reward shaping constants
    REWARD_GOAL: float = 1.0
    REWARD_OBSTACLE: float = -1.0
    REWARD_STEP: float = -0.01   # small living penalty to encourage efficiency

    NUM_ACTIONS: int = 4

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        num_obstacles: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the grid world.

        Args:
            grid_size:      Side length of the square grid.
            num_obstacles:  Number of randomly placed obstacles.
            seed:           Optional RNG seed for reproducibility.
        """
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.goal: Tuple[int, int] = (grid_size - 1, grid_size - 1)
        self.start: Tuple[int, int] = (0, 0)

        self.obstacles: List[Tuple[int, int]] = []
        self.agent_pos: Tuple[int, int] = self.start
        self.steps: int = 0
        self.done: bool = False

        self._place_obstacles()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """Reset environment to the initial state.

        Returns:
            Initial observation (flat index of starting cell).
        """
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self._place_obstacles()
        return self._obs()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Apply action and return (next_state, reward, done, info).

        Args:
            action: Integer in [0, 3].

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Episode is done — call reset() before stepping.")

        self.steps += 1
        row, col = self.agent_pos

        # Compute candidate next position
        if action == 0:    # UP
            row -= 1
        elif action == 1:  # DOWN
            row += 1
        elif action == 2:  # LEFT
            col -= 1
        elif action == 3:  # RIGHT
            col += 1
        else:
            raise ValueError(f"Invalid action {action}. Must be in [0, 3].")

        # Clamp to grid boundaries (walls — agent stays in place)
        row = max(0, min(self.grid_size - 1, row))
        col = max(0, min(self.grid_size - 1, col))

        new_pos = (row, col)

        # Determine reward
        if new_pos == self.goal:
            reward = self.REWARD_GOAL
            self.done = True
        elif new_pos in self.obstacles:
            reward = self.REWARD_OBSTACLE
            self.done = True
        else:
            reward = self.REWARD_STEP

        # Enforce episode time-limit
        if self.steps >= MAX_STEPS:
            self.done = True

        self.agent_pos = new_pos
        info = {"steps": self.steps}
        return self._obs(), reward, self.done, info

    def render(self) -> None:
        """Print an ASCII representation of the current grid state."""
        lines: List[str] = []
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                cell = (r, c)
                if cell == self.agent_pos:
                    row_str += " A "
                elif cell == self.goal:
                    row_str += " G "
                elif cell in self.obstacles:
                    row_str += " X "
                else:
                    row_str += " . "
            lines.append(row_str)
        print("\n".join(lines))
        print()

    def get_state(self) -> int:
        """Return the current observation."""
        return self._obs()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self) -> int:
        """Convert (row, col) to a flat integer index."""
        row, col = self.agent_pos
        return row * self.grid_size + col

    def _place_obstacles(self) -> None:
        """Randomly place obstacles, avoiding start and goal cells."""
        forbidden = {self.start, self.goal}
        candidates = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in forbidden
        ]
        self.obstacles = self.rng.sample(candidates, min(self.num_obstacles, len(candidates)))

    @property
    def state_size(self) -> int:
        """Total number of distinct states (grid cells)."""
        return self.grid_size * self.grid_size
