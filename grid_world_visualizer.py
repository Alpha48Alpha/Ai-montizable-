#!/usr/bin/env python3
"""
Grid World Visualizer
=====================
Matplotlib-based visualization module that displays an agent moving through
a grid world during training / simulation.

Features
--------
- Configurable grid size, start position, and goal position.
- Live display: agent position and goal location are rendered each step.
- Step counter shown in the plot title.
- Non-blocking animation via ``plt.pause()`` so every step is visible.
- Headless-safe: falls back to the ``Agg`` backend when no display is
  available (e.g. CI servers).

Run:
    python grid_world_visualizer.py
"""

import random
import sys

# ---------------------------------------------------------------------------
# Matplotlib setup — use a non-interactive backend when there is no display
# ---------------------------------------------------------------------------
import matplotlib
try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.figure()          # test whether a display is available
    plt.close()
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------

class GridWorld:
    """A simple 2-D grid world with an agent and a goal cell.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (default 5×5).
    start : tuple[int, int] | None
        Initial ``(row, col)`` for the agent.  ``None`` → top-left corner.
    goal : tuple[int, int] | None
        ``(row, col)`` of the goal cell.  ``None`` → bottom-right corner.
    """

    ACTIONS = {
        "up":    (-1,  0),
        "down":  ( 1,  0),
        "left":  ( 0, -1),
        "right": ( 0,  1),
    }

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        start: "tuple[int, int] | None" = None,
        goal:  "tuple[int, int] | None" = None,
    ) -> None:
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2×2.")
        self.rows = rows
        self.cols = cols
        self._start: tuple[int, int] = start if start is not None else (0, 0)
        self._goal:  tuple[int, int] = goal  if goal  is not None else (rows - 1, cols - 1)

        if not self._in_bounds(self._start):
            raise ValueError(f"Start position {self._start} is outside the grid.")
        if not self._in_bounds(self._goal):
            raise ValueError(f"Goal position {self._goal} is outside the grid.")
        if self._start == self._goal:
            raise ValueError("Start and goal must be different cells.")

        self.agent: tuple[int, int] = self._start
        self.step_count: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    def reset(self) -> "tuple[int, int]":
        """Reset the environment to the initial state."""
        self.agent = self._start
        self.step_count = 0
        self.done = False
        return self.agent

    def step(self, action: str) -> "tuple[tuple[int, int], float, bool]":
        """Apply *action* and return ``(new_state, reward, done)``.

        Parameters
        ----------
        action : str
            One of ``"up"``, ``"down"``, ``"left"``, ``"right"``.

        Returns
        -------
        state : tuple[int, int]
            New agent position ``(row, col)``.
        reward : float
            ``+1.0`` on reaching the goal, ``-0.01`` per step otherwise.
        done : bool
            ``True`` when the agent has reached the goal.
        """
        if self.done:
            raise RuntimeError("Episode is over — call reset() first.")
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Choose from {list(self.ACTIONS)}.")

        dr, dc = self.ACTIONS[action]
        new_r = max(0, min(self.rows - 1, self.agent[0] + dr))
        new_c = max(0, min(self.cols - 1, self.agent[1] + dc))
        self.agent = (new_r, new_c)
        self.step_count += 1

        if self.agent == self._goal:
            self.done = True
            reward = 1.0
        else:
            reward = -0.01

        return self.agent, reward, self.done

    # ------------------------------------------------------------------
    def _in_bounds(self, pos: "tuple[int, int]") -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    @property
    def goal(self) -> "tuple[int, int]":
        return self._goal


# ---------------------------------------------------------------------------
# GridWorldVisualizer
# ---------------------------------------------------------------------------

class GridWorldVisualizer:
    """Matplotlib visualizer for a :class:`GridWorld`.

    Parameters
    ----------
    env : GridWorld
        The environment to visualize.
    pause : float
        Seconds to pause between frames (controls animation speed).
    """

    _AGENT_COLOR = "#2196F3"   # blue
    _GOAL_COLOR  = "#4CAF50"   # green
    _GRID_COLOR  = "#ECEFF1"   # light grey cell background
    _BORDER_COLOR = "#90A4AE"  # slate cell borders

    def __init__(self, env: GridWorld, pause: float = 0.3) -> None:
        self.env = env
        self.pause = pause

        self._fig, self._ax = plt.subplots(figsize=(max(4, env.cols), max(4, env.rows)))
        self._fig.patch.set_facecolor("#FAFAFA")
        self._ax.set_aspect("equal")
        self._ax.set_xlim(0, env.cols)
        self._ax.set_ylim(0, env.rows)
        self._ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self._ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        self._ax.tick_params(left=False, bottom=False,
                             labelleft=False, labelbottom=False)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(self._BORDER_COLOR)

        self._draw_grid()
        self._agent_patch = self._make_circle(env.agent, self._AGENT_COLOR, zorder=3)
        self._goal_patch  = self._make_circle(env.goal,  self._GOAL_COLOR,  zorder=2)

        legend = [
            mpatches.Patch(color=self._AGENT_COLOR, label="Agent"),
            mpatches.Patch(color=self._GOAL_COLOR,  label="Goal"),
        ]
        self._ax.legend(handles=legend, loc="upper right",
                        fontsize=8, framealpha=0.9)

        self._update_title()
        plt.tight_layout()
        plt.ion()
        self._fig.canvas.draw()
        plt.pause(self.pause)

    # ------------------------------------------------------------------
    def update(self) -> None:
        """Redraw the visualizer to reflect the current environment state."""
        self._move_circle(self._agent_patch, self.env.agent)
        self._update_title()
        self._fig.canvas.draw()
        plt.pause(self.pause)

    def close(self) -> None:
        """Close the matplotlib figure."""
        plt.close(self._fig)

    # ------------------------------------------------------------------
    def _draw_grid(self) -> None:
        rows, cols = self.env.rows, self.env.cols
        for r in range(rows):
            for c in range(cols):
                rect = mpatches.FancyBboxPatch(
                    (c + 0.05, r + 0.05),
                    0.90, 0.90,
                    boxstyle="round,pad=0.02",
                    linewidth=1,
                    edgecolor=self._BORDER_COLOR,
                    facecolor=self._GRID_COLOR,
                    zorder=1,
                )
                self._ax.add_patch(rect)

    def _cell_center(self, pos: "tuple[int, int]") -> "tuple[float, float]":
        """Convert grid ``(row, col)`` to matplotlib ``(x, y)`` coordinates.

        Matplotlib's y-axis goes upward, so row 0 is at the *top* of the
        rendered grid — this matches the conventional matrix layout.
        """
        row, col = pos
        x = col + 0.5
        y = (self.env.rows - 1 - row) + 0.5
        return x, y

    def _make_circle(
        self, pos: "tuple[int, int]", color: str, zorder: int = 2
    ) -> mpatches.Circle:
        x, y = self._cell_center(pos)
        circle = mpatches.Circle((x, y), radius=0.35,
                                 color=color, zorder=zorder)
        self._ax.add_patch(circle)
        return circle

    def _move_circle(
        self, circle: mpatches.Circle, pos: "tuple[int, int]"
    ) -> None:
        x, y = self._cell_center(pos)
        circle.center = (x, y)

    def _update_title(self) -> None:
        status = "✓ Goal reached!" if self.env.done else "running …"
        self._ax.set_title(
            f"Grid World  |  step {self.env.step_count}  |  {status}",
            fontsize=10,
            pad=8,
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: GridWorld,
    viz: "GridWorldVisualizer | None" = None,
    max_steps: int = 200,
    policy: "callable | None" = None,
) -> "dict[str, object]":
    """Run one episode, optionally visualizing each step.

    Parameters
    ----------
    env : GridWorld
        The environment (will be reset at the start of the episode).
    viz : GridWorldVisualizer | None
        If provided, :meth:`GridWorldVisualizer.update` is called after
        every step so the display is refreshed in real time.
    max_steps : int
        Hard limit on episode length.
    policy : callable | None
        A callable ``policy(state) → action``.  Defaults to a uniform
        random policy over all four actions.

    Returns
    -------
    dict
        ``{"steps": int, "total_reward": float, "reached_goal": bool}``
    """
    actions = list(GridWorld.ACTIONS.keys())
    if policy is None:
        def policy(_state):
            return random.choice(actions)

    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = policy(state)
        state, reward, done = env.step(action)
        total_reward += reward

        if viz is not None:
            viz.update()

        if done:
            break

    return {
        "steps": env.step_count,
        "total_reward": round(total_reward, 4),
        "reached_goal": env.done,
    }


# ---------------------------------------------------------------------------
# Main — demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the visualizer with a random-walk agent on a 6×6 grid."""
    print("\n🗺   Grid World Visualizer\n")

    env = GridWorld(rows=6, cols=6)
    viz = GridWorldVisualizer(env, pause=0.2)

    random.seed(42)
    result = run_episode(env, viz=viz, max_steps=300)

    print(f"  Steps taken  : {result['steps']}")
    print(f"  Total reward : {result['total_reward']}")
    print(f"  Reached goal : {result['reached_goal']}")

    # Keep the final frame visible for a moment before closing
    plt.pause(1.5)
    viz.close()
    print("\n✅  Visualization complete.")


if __name__ == "__main__":
    main()
