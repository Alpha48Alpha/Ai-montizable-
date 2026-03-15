"""
simple_world.py — Alpha48Alpha AI Lab
======================================
A minimal grid-world environment compatible with the OpenAI Gym interface.

The agent starts at position 0 on a 1-D grid and must reach the goal at
position (grid_size - 1) within a fixed number of steps.

Observation : current position normalised to [0, 1]  (shape: [1])
Action space: 0 = move left, 1 = move right           (discrete, 2 actions)
Reward      : +1.0 when the goal is reached, -0.01 per step (small time penalty)
Episode ends: when the goal is reached OR max_steps is exceeded
"""


class SimpleWorld:
    """
    A 1-D grid world with a single agent and a single goal cell.

    Parameters
    ----------
    grid_size : int
        Total number of cells.  The goal is always the last cell.
    max_steps : int
        Maximum number of steps before the episode is truncated.
    """

    # Number of discrete actions available to the agent
    N_ACTIONS: int = 2
    # Dimensionality of the observation vector returned to the agent
    OBS_DIM: int = 1

    def __init__(self, grid_size: int = 10, max_steps: int = 100) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        self.grid_size = grid_size
        self.max_steps = max_steps

        # These are set (or reset) by reset()
        self._position: int = 0
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> list:
        """
        Reset the environment to the initial state.

        Returns
        -------
        observation : list[float]
            Normalised agent position [pos / (grid_size - 1)].
        """
        self._position = 0
        self._steps = 0
        return self._observe()

    def step(self, action: int) -> tuple:
        """
        Apply *action* and advance the world by one time-step.

        Parameters
        ----------
        action : int
            0 → move left (clamped at 0), 1 → move right (clamped at grid_size - 1).

        Returns
        -------
        observation : list[float]
            New normalised position.
        reward : float
            +1.0 at the goal, -0.01 otherwise.
        done : bool
            True when the episode is over (goal reached or max_steps hit).
        info : dict
            Auxiliary diagnostics (step count, raw position).
        """
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action}. Expected 0 or 1.")

        # Move agent; clamp to valid grid range
        if action == 1:
            self._position = min(self._position + 1, self.grid_size - 1)
        else:
            self._position = max(self._position - 1, 0)

        self._steps += 1

        # Check terminal conditions
        goal_reached = self._position == self.grid_size - 1
        timeout = self._steps >= self.max_steps

        reward = 1.0 if goal_reached else -0.01
        done = goal_reached or timeout

        info = {
            "steps": self._steps,
            "position": self._position,
            "goal_reached": goal_reached,
        }

        return self._observe(), reward, done, info

    def render(self) -> str:
        """
        Return a simple ASCII representation of the current grid state.

        Returns
        -------
        str
            A string like ``|_A_____G|`` where A = agent, G = goal.
        """
        cells = ["_"] * self.grid_size
        cells[self.grid_size - 1] = "G"  # goal marker (may be overwritten)
        cells[self._position] = "A"       # agent marker
        return "|" + "".join(cells) + "|"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _observe(self) -> list:
        """Return the normalised position as a single-element list."""
        return [self._position / (self.grid_size - 1)]
