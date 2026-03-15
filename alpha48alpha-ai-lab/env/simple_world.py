"""
Simple 2D Grid World Environment
=================================
A minimal simulated environment for reinforcement learning experiments.

The agent navigates a 2D grid aiming to reach a goal cell while
avoiding obstacle cells.

Grid cell values:
    0  — empty cell (free to walk on)
    1  — obstacle   (blocked; stepping here gives a penalty)
    2  — goal       (stepping here gives a positive reward)
    3  — agent      (current position; displayed during render)
"""

import random


class SimpleWorld:
    """2D grid world compatible with reinforcement learning training loops."""

    # Reward constants
    REWARD_GOAL = 1.0
    REWARD_OBSTACLE = -0.5
    REWARD_STEP = -0.01  # small living penalty to encourage short paths

    # Action mapping: 0=up, 1=down, 2=left, 3=right
    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, grid_size: int = 5, obstacle_ratio: float = 0.15,
                 seed: int | None = None):
        """
        Parameters
        ----------
        grid_size     : Side length of the square grid.
        obstacle_ratio: Fraction of non-start/non-goal cells to block.
        seed          : Optional random seed for reproducibility.
        """
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self._rng = random.Random(seed)
        self._state_size = grid_size * grid_size

        # Initialise grid and agent
        self._grid: list[list[int]] = []
        self._agent_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (grid_size - 1, grid_size - 1)
        self._done: bool = False

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> list[float]:
        """Reset the environment to an initial state.

        Returns
        -------
        list[float]
            Flat one-hot observation of the grid (agent position encoded).
        """
        self._done = False
        self._agent_pos = (0, 0)
        self._goal_pos = (self.grid_size - 1, self.grid_size - 1)

        # Build an empty grid
        self._grid = [[0] * self.grid_size for _ in range(self.grid_size)]

        # Place random obstacles (never on start or goal)
        total_cells = self.grid_size * self.grid_size
        num_obstacles = int(total_cells * self.obstacle_ratio)
        candidates = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in (self._agent_pos, self._goal_pos)
        ]
        obstacles = self._rng.sample(candidates, min(num_obstacles, len(candidates)))
        for r, c in obstacles:
            self._grid[r][c] = 1  # mark as obstacle

        return self._get_observation()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        """Apply an action and return the next observation, reward, and done flag.

        Parameters
        ----------
        action : int
            Integer in [0, NUM_ACTIONS).

        Returns
        -------
        observation : list[float]
            New flat observation after taking the action.
        reward : float
            Reward received for this transition.
        done : bool
            True if the episode has ended (goal reached or invalid move limit).
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() before stepping.")

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be in {list(self.ACTIONS)}.")

        dr, dc = self.ACTIONS[action]
        nr, nc = self._agent_pos[0] + dr, self._agent_pos[1] + dc

        # Check grid boundaries
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            # Stay in place; penalise wall bump like an obstacle
            reward = self.REWARD_OBSTACLE
            return self._get_observation(), reward, self._done

        cell = self._grid[nr][nc]

        if cell == 1:
            # Obstacle: stay in place, apply penalty
            reward = self.REWARD_OBSTACLE
        elif (nr, nc) == self._goal_pos:
            # Goal reached
            self._agent_pos = (nr, nc)
            reward = self.REWARD_GOAL
            self._done = True
        else:
            # Normal empty cell
            self._agent_pos = (nr, nc)
            reward = self.REWARD_STEP

        return self._get_observation(), reward, self._done

    def render(self) -> None:
        """Print a simple ASCII representation of the current grid state."""
        symbols = {0: ".", 1: "#", 2: "G"}
        print(f"  {'_' * (self.grid_size * 2 + 1)}")
        for r in range(self.grid_size):
            row_str = "| "
            for c in range(self.grid_size):
                if (r, c) == self._agent_pos and not self._done:
                    row_str += "A "
                elif (r, c) == self._goal_pos:
                    row_str += "G "
                else:
                    row_str += symbols.get(self._grid[r][c], "?") + " "
            row_str += "|"
            print(f"  {row_str}")
        print(f"  {'‾' * (self.grid_size * 2 + 1)}")
        print(f"  Agent: {self._agent_pos}  Goal: {self._goal_pos}  Done: {self._done}\n")

    @property
    def state_size(self) -> int:
        """Number of features in the flat observation vector."""
        return self._state_size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> list[float]:
        """Return a flat one-hot vector with a 1.0 at the agent's cell index."""
        obs = [0.0] * self._state_size
        r, c = self._agent_pos
        obs[r * self.grid_size + c] = 1.0
        return obs
