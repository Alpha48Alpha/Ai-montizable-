"""
simple_world.py — Alpha48Alpha AI Lab
======================================
A 2-D grid world environment compatible with the OpenAI Gym-style interface.

The agent starts at the top-left cell (0, 0) and must navigate to the
bottom-right goal cell (rows-1, cols-1) while avoiding obstacle cells.

Observation : normalised (row, col) of the agent  → shape [2],  values in [0, 1]
Action space: 0 = up, 1 = down, 2 = left, 3 = right              (discrete, 4 actions)
Reward      : +1.0  goal reached
              -0.5  attempted move into an obstacle (agent stays put)
              -0.01 every other step (time penalty encouraging efficiency)
Episode ends: goal reached  OR  max_steps exceeded
"""

import random


# Map action index → (row_delta, col_delta)
_ACTION_DELTAS = {
    0: (-1,  0),   # up
    1: ( 1,  0),   # down
    2: ( 0, -1),   # left
    3: ( 0,  1),   # right
}

# Human-readable action labels used by render helpers
ACTION_LABELS = {0: "U", 1: "D", 2: "L", 3: "R"}


class SimpleWorld:
    """
    A 2-D grid world with obstacles, a single agent, and a single goal cell.

    Parameters
    ----------
    rows : int
        Number of rows in the grid (must be ≥ 2).
    cols : int
        Number of columns in the grid (must be ≥ 2).
    max_steps : int
        Maximum number of steps before the episode is truncated.
    obstacle_density : float
        Fraction of non-start, non-goal cells to fill with obstacles.
        Clamped to [0.0, 0.8] to keep the grid solvable.
    seed : int or None
        Optional random seed for reproducible obstacle placement.
    """

    # Number of discrete actions available to the agent
    N_ACTIONS: int = 4
    # Dimensionality of the observation vector returned to the agent
    OBS_DIM: int = 2

    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        max_steps: int = 200,
        obstacle_density: float = 0.15,
        seed: int | None = None,
    ) -> None:
        if rows < 2 or cols < 2:
            raise ValueError("rows and cols must each be at least 2.")
        if not 0.0 <= obstacle_density <= 0.8:
            raise ValueError("obstacle_density must be between 0.0 and 0.8.")

        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density

        # Fixed start and goal positions
        self._start = (0, 0)
        self._goal = (rows - 1, cols - 1)

        # Build the static obstacle layout once (reused across episodes)
        self._rng = random.Random(seed)
        self._obstacles: set[tuple[int, int]] = self._place_obstacles()

        # Dynamic state — initialised by reset()
        self._agent: tuple[int, int] = self._start
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> list:
        """
        Reset the agent to the start cell.

        Returns
        -------
        observation : list[float]
            Normalised [row, col] of the agent, each in [0, 1].
        """
        self._agent = self._start
        self._steps = 0
        return self._observe()

    def step(self, action: int) -> tuple:
        """
        Apply *action* and advance the world by one time-step.

        If the intended move would leave the grid boundary, the agent
        stays in place (treated as an invalid move, no obstacle penalty).
        If the intended move leads into an obstacle, the agent stays put
        and receives the obstacle penalty.

        Parameters
        ----------
        action : int
            One of: 0 (up), 1 (down), 2 (left), 3 (right).

        Returns
        -------
        observation : list[float]
            New normalised [row, col].
        reward : float
            +1.0 at goal, -0.5 for obstacle collision, -0.01 otherwise.
        done : bool
            True when the episode is over.
        info : dict
            Auxiliary diagnostics.
        """
        if action not in _ACTION_DELTAS:
            raise ValueError(
                f"Invalid action {action}. Expected one of {list(_ACTION_DELTAS)}."
            )

        dr, dc = _ACTION_DELTAS[action]
        new_row = self._agent[0] + dr
        new_col = self._agent[1] + dc

        # Determine reward and whether the agent actually moves
        hit_obstacle = False
        if not self._in_bounds(new_row, new_col):
            # Wall — stay put, small time penalty (not an obstacle hit)
            reward = -0.01
        elif (new_row, new_col) in self._obstacles:
            # Obstacle — stay put, heavier penalty
            hit_obstacle = True
            reward = -0.5
        else:
            # Valid move — update position
            self._agent = (new_row, new_col)
            reward = -0.01

        self._steps += 1

        # Check terminal conditions
        goal_reached = self._agent == self._goal
        timeout = self._steps >= self.max_steps

        if goal_reached:
            reward = 1.0

        done = goal_reached or timeout

        info = {
            "steps": self._steps,
            "position": self._agent,
            "goal_reached": goal_reached,
            "hit_obstacle": hit_obstacle,
        }

        return self._observe(), reward, done, info

    def render(self) -> str:
        """
        Return a multi-line ASCII representation of the current grid state.

        Legend
        ------
        A  agent position
        G  goal  (or *  if agent is on the goal)
        #  obstacle
        .  empty cell

        Returns
        -------
        str
            A grid string with a top/bottom border.
        """
        lines = ["+" + "-" * self.cols + "+"]
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                cell = (r, c)
                if cell == self._agent and cell == self._goal:
                    row_str += "*"   # agent reached goal
                elif cell == self._agent:
                    row_str += "A"
                elif cell == self._goal:
                    row_str += "G"
                elif cell in self._obstacles:
                    row_str += "#"
                else:
                    row_str += "."
            row_str += "|"
            lines.append(row_str)
        lines.append("+" + "-" * self.cols + "+")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is a valid grid cell."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _place_obstacles(self) -> set[tuple[int, int]]:
        """
        Randomly select obstacle cells.

        Start and goal cells are always kept clear.  The total number of
        obstacle cells is floor(obstacle_density * (rows * cols - 2)).
        """
        # All cells except start and goal are candidates
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in (self._start, self._goal)
        ]
        n_obstacles = int(self.obstacle_density * len(candidates))
        chosen = self._rng.sample(candidates, n_obstacles)
        return set(chosen)

    def _observe(self) -> list:
        """
        Return the normalised agent position.

        Each coordinate is divided by (dimension - 1) so values lie in [0, 1].
        """
        norm_row = self._agent[0] / (self.rows - 1)
        norm_col = self._agent[1] / (self.cols - 1)
        return [norm_row, norm_col]
