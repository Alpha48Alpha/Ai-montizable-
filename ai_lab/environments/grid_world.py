"""Simple grid-world simulation environment."""

from typing import Tuple

from ai_lab.environments.base_env import BaseEnv


class GridWorld(BaseEnv):
    """An N×N grid world where an agent navigates to a goal.

    States are encoded as a single integer ``row * size + col``.

    Actions
    -------
    0 — up
    1 — right
    2 — down
    3 — left

    Rewards
    -------
    +1.0  on reaching the goal (bottom-right corner)
    -0.01 every other step (small living penalty to encourage efficiency)
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def __init__(self, size: int = 5) -> None:
        if size < 2:
            raise ValueError("Grid size must be at least 2.")
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self._row = 0
        self._col = 0

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> int:
        self._row, self._col = 0, 0
        return self._encode(self._row, self._col)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be 0–3.")
        dr, dc = self.ACTIONS[action]
        new_row = max(0, min(self.size - 1, self._row + dr))
        new_col = max(0, min(self.size - 1, self._col + dc))
        self._row, self._col = new_row, new_col

        done = (self._row == self.size - 1 and self._col == self.size - 1)
        reward = 1.0 if done else -0.01
        obs = self._encode(self._row, self._col)
        return obs, reward, done, {"row": self._row, "col": self._col}

    def render(self) -> str:
        lines = []
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if r == self._row and c == self._col:
                    row_str += " A "
                elif r == self.size - 1 and c == self.size - 1:
                    row_str += " G "
                else:
                    row_str += " . "
            lines.append(row_str)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, row: int, col: int) -> int:
        return row * self.size + col
