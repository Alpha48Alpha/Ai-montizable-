"""
environments/grid_world.py — Simple N×N grid-world environment.

The grid is an N×N matrix.  The agent starts at the top-left corner
(0, 0) and must navigate to the goal at the bottom-right corner
(N-1, N-1).  No obstacles are present — this is the simplest possible
environment for verifying a new agent implementation.

Action space (discrete, 4 actions)
-----------------------------------
  0 — UP    (row - 1)
  1 — DOWN  (row + 1)
  2 — LEFT  (col - 1)
  3 — RIGHT (col + 1)

State representation
--------------------
A flat float vector of length 2: [row / (N-1), col / (N-1)].
Values are normalised to [0, 1] so that neural networks receive
inputs of consistent scale regardless of grid size.
"""

from __future__ import annotations

import numpy as np


# Action index → (delta_row, delta_col)
_ACTION_DELTAS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}


class GridWorld:
    """
    Minimal N×N grid-world compatible with the Alpha48Alpha training loop.

    Parameters
    ----------
    size : int
        Side length of the square grid (default: 8 → 8×8 grid).
    reward_goal : float
        Reward given when the agent reaches the goal cell.
    reward_step : float
        Reward (usually negative) given for every non-terminal step.
    max_steps : int
        Episode terminates after this many steps even if the goal is
        not reached.
    """

    def __init__(
        self,
        size: int = 8,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        max_steps: int = 200,
    ):
        self.size        = size
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.max_steps   = max_steps

        # Agent & goal positions (set by reset())
        self._agent_row: int = 0
        self._agent_col: int = 0
        self._goal_row:  int = size - 1
        self._goal_col:  int = size - 1
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.

        Returns
        -------
        np.ndarray, shape (2,)
            Normalised [row, col] of the agent.
        """
        self._agent_row  = 0
        self._agent_col  = 0
        self._step_count = 0
        return self.get_state()

    def step(self, action: int):
        """
        Apply an action and advance the environment by one time-step.

        Parameters
        ----------
        action : int
            One of {0, 1, 2, 3} corresponding to UP / DOWN / LEFT / RIGHT.

        Returns
        -------
        next_state : np.ndarray, shape (2,)
        reward : float
        done : bool
        info : dict
        """
        if action not in _ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Must be in {{0,1,2,3}}.")

        dr, dc = _ACTION_DELTAS[action]
        new_row = np.clip(self._agent_row + dr, 0, self.size - 1)
        new_col = np.clip(self._agent_col + dc, 0, self.size - 1)

        self._agent_row  = int(new_row)
        self._agent_col  = int(new_col)
        self._step_count += 1

        # Check terminal conditions
        reached_goal = (
            self._agent_row == self._goal_row
            and self._agent_col == self._goal_col
        )
        timeout = self._step_count >= self.max_steps

        if reached_goal:
            reward = self.reward_goal
            done   = True
        else:
            reward = self.reward_step
            done   = timeout

        return self.get_state(), reward, done, {"steps": self._step_count}

    def render(self) -> None:
        """Print a simple ASCII representation of the current grid."""
        print(f"\n--- GridWorld {self.size}×{self.size} | Step {self._step_count} ---")
        for row in range(self.size):
            row_str = ""
            for col in range(self.size):
                if row == self._agent_row and col == self._agent_col:
                    row_str += " A "   # Agent
                elif row == self._goal_row and col == self._goal_col:
                    row_str += " G "   # Goal
                else:
                    row_str += " . "
            print(row_str)
        print()

    def get_state(self) -> np.ndarray:
        """
        Return the current normalised state vector.

        Returns
        -------
        np.ndarray, shape (2,)  — [row/(N-1), col/(N-1)]
        """
        norm = float(self.size - 1) if self.size > 1 else 1.0
        return np.array(
            [self._agent_row / norm, self._agent_col / norm],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """Dimensionality of the observation vector."""
        return 2

    @property
    def action_size(self) -> int:
        """Number of discrete actions."""
        return 4
