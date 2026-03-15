#!/usr/bin/env python3
"""
RL World Simulation Environment
================================
A 2D grid-world environment for reinforcement learning.

Features:
  - Configurable grid size
  - Agent navigation (4-directional movement)
  - Randomly placed obstacles
  - One or more goal states
  - Rewards and penalties

Compatible interface:
  env = WorldSimEnv()
  state = env.reset()
  state, reward, done, info = env.step(action)
  env.render()
  state = env.get_state()

Actions:
  0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT

Run a quick demo:
  python rl_environment.py
"""

import random


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

REWARD_GOAL       =  10.0   # reaching a goal cell
REWARD_STEP       =  -0.1   # small penalty per step to encourage efficiency
REWARD_OBSTACLE   =  -1.0   # penalty for trying to walk into an obstacle
REWARD_OUT_BOUNDS =  -1.0   # penalty for trying to leave the grid

# Cell type codes used in the grid
CELL_EMPTY    = 0
CELL_OBSTACLE = 1
CELL_GOAL     = 2
CELL_AGENT    = 3   # used only for rendering

# Render symbols
_SYMBOLS = {
    CELL_EMPTY:    ".",
    CELL_OBSTACLE: "#",
    CELL_GOAL:     "G",
    CELL_AGENT:    "A",
}


# ---------------------------------------------------------------------------
# WorldSimEnv
# ---------------------------------------------------------------------------

class WorldSimEnv:
    """2D grid-world environment for reinforcement learning.

    Parameters
    ----------
    width : int
        Number of columns in the grid (default 8).
    height : int
        Number of rows in the grid (default 8).
    num_obstacles : int
        Number of obstacle cells randomly placed at each reset (default 10).
    num_goals : int
        Number of goal cells randomly placed at each reset (default 1).
    max_steps : int
        Maximum steps per episode before forced termination (default 200).
    seed : int or None
        Random seed for reproducibility.  Pass ``None`` for non-deterministic.
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        num_obstacles: int = 10,
        num_goals: int = 1,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        if width < 3 or height < 3:
            raise ValueError("Grid must be at least 3×3.")
        if num_obstacles + num_goals >= width * height - 1:
            raise ValueError(
                "Too many obstacles/goals for the given grid size."
            )

        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.num_goals = num_goals
        self.max_steps = max_steps

        self._rng = random.Random(seed)

        # Internal state (populated by reset)
        self._grid: list[list[int]] = []
        self._agent_pos: tuple[int, int] = (0, 0)
        self._goal_positions: list[tuple[int, int]] = []
        self._steps: int = 0
        self._done: bool = False

        # Initialise so the env is usable before the first explicit reset()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> list[int]:
        """Reset the environment to a new random episode.

        Returns
        -------
        list[int]
            Flat state vector (see :meth:`get_state`).
        """
        self._steps = 0
        self._done = False
        self._grid = [
            [CELL_EMPTY] * self.width for _ in range(self.height)
        ]

        # Collect all cell coordinates and shuffle them
        all_cells: list[tuple[int, int]] = [
            (r, c) for r in range(self.height) for c in range(self.width)
        ]
        self._rng.shuffle(all_cells)

        # First cell → agent start position
        self._agent_pos = all_cells[0]
        idx = 1

        # Next num_goals cells → goals
        self._goal_positions = []
        for i in range(self.num_goals):
            pos = all_cells[idx + i]
            self._goal_positions.append(pos)
            self._grid[pos[0]][pos[1]] = CELL_GOAL
        idx += self.num_goals

        # Next num_obstacles cells → obstacles
        for i in range(self.num_obstacles):
            pos = all_cells[idx + i]
            self._grid[pos[0]][pos[1]] = CELL_OBSTACLE

        return self.get_state()

    def step(self, action: int) -> tuple[list[int], float, bool, dict]:
        """Execute one action in the environment.

        Parameters
        ----------
        action : int
            One of ``ACTION_UP`` (0), ``ACTION_DOWN`` (1),
            ``ACTION_LEFT`` (2), ``ACTION_RIGHT`` (3).

        Returns
        -------
        state : list[int]
            New flat state vector.
        reward : float
            Reward received for this transition.
        done : bool
            ``True`` if the episode has ended.
        info : dict
            Auxiliary diagnostic information.
        """
        if self._done:
            raise RuntimeError(
                "Episode has ended. Call reset() before stepping again."
            )
        if action not in ACTIONS:
            raise ValueError(
                f"Invalid action {action!r}. Must be one of {ACTIONS}."
            )

        self._steps += 1
        row, col = self._agent_pos

        delta = {
            ACTION_UP:    (-1,  0),
            ACTION_DOWN:  ( 1,  0),
            ACTION_LEFT:  ( 0, -1),
            ACTION_RIGHT: ( 0,  1),
        }[action]

        new_row = row + delta[0]
        new_col = col + delta[1]

        reward = REWARD_STEP
        info: dict = {"step": self._steps}

        # Check boundary collision
        if not (0 <= new_row < self.height and 0 <= new_col < self.width):
            reward += REWARD_OUT_BOUNDS
            info["event"] = "out_of_bounds"
        # Check obstacle collision
        elif self._grid[new_row][new_col] == CELL_OBSTACLE:
            reward += REWARD_OBSTACLE
            info["event"] = "obstacle"
        else:
            # Valid move — update agent position
            self._agent_pos = (new_row, new_col)
            row, col = new_row, new_col

            if self._grid[row][col] == CELL_GOAL:
                reward += REWARD_GOAL
                info["event"] = "goal_reached"
                self._done = True
            else:
                info["event"] = "move"

        # Check step limit
        if self._steps >= self.max_steps:
            self._done = True
            info["timeout"] = True

        return self.get_state(), reward, self._done, info

    def render(self) -> None:
        """Print an ASCII representation of the current grid to stdout.

        Legend:
          ``.``  empty cell
          ``#``  obstacle
          ``G``  goal
          ``A``  agent
        """
        print(f"  Steps: {self._steps} / {self.max_steps}")
        print("  +" + "-" * (self.width * 2 - 1) + "+")
        for r in range(self.height):
            row_chars: list[str] = []
            for c in range(self.width):
                if (r, c) == self._agent_pos:
                    row_chars.append(_SYMBOLS[CELL_AGENT])
                else:
                    row_chars.append(_SYMBOLS[self._grid[r][c]])
            print("  |" + " ".join(row_chars) + "|")
        print("  +" + "-" * (self.width * 2 - 1) + "+")
        print(f"  Agent: {self._agent_pos}  |  Done: {self._done}")

    def get_state(self) -> list[int]:
        """Return a flat representation of the current environment state.

        The state vector contains:

        * ``agent_row`` — current row of the agent (int)
        * ``agent_col`` — current column of the agent (int)
        * Flattened grid values (row-major order), one int per cell.
          Cell values: 0 = empty, 1 = obstacle, 2 = goal.

        Returns
        -------
        list[int]
            A list of length ``2 + width * height``.
        """
        agent_row, agent_col = self._agent_pos
        flat_grid: list[int] = []
        for row in self._grid:
            flat_grid.extend(row)
        return [agent_row, agent_col] + flat_grid

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_size(self) -> int:
        """Length of the state vector returned by :meth:`get_state`."""
        return 2 + self.width * self.height

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions available."""
        return len(ACTIONS)

    @property
    def steps(self) -> int:
        """Number of steps taken in the current episode."""
        return self._steps

    @property
    def is_done(self) -> bool:
        """Whether the current episode has ended."""
        return self._done


# ---------------------------------------------------------------------------
# Demo / quick-test training loop
# ---------------------------------------------------------------------------

def _random_policy(action_space_size: int) -> int:
    """Random policy: choose a uniformly random action."""
    return random.randrange(action_space_size)


def run_demo(
    episodes: int = 3,
    render: bool = True,
    width: int = 8,
    height: int = 8,
    num_obstacles: int = 10,
    num_goals: int = 1,
    max_steps: int = 50,
    seed: int | None = 42,
) -> None:
    """Run a short random-policy demo to verify the environment."""
    env = WorldSimEnv(
        width=width,
        height=height,
        num_obstacles=num_obstacles,
        num_goals=num_goals,
        max_steps=max_steps,
        seed=seed,
    )

    print("=" * 60)
    print("  RL World Simulation Environment — Demo")
    print(f"  Grid: {env.width}×{env.height}  |  "
          f"Obs: {env.num_obstacles}  |  Goals: {env.num_goals}")
    print(f"  State size: {env.observation_size}  |  "
          f"Actions: {env.action_space_size}")
    print("=" * 60)

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        print(f"\n--- Episode {ep} ---")
        if render:
            env.render()

        while not done:
            action = _random_policy(env.action_space_size)
            state, reward, done, info = env.step(action)
            total_reward += reward

        print(f"\n  Episode {ep} finished.")
        print(f"  Total reward : {total_reward:.2f}")
        print(f"  Steps taken  : {env.steps}")
        print(f"  Final event  : {info.get('event', 'N/A')}")
        if render:
            env.render()

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
