"""
visualization.py — Matplotlib visualization for grid world training.

Provides two utilities:
    - ``GridWorldVisualizer``: real-time or save-to-file grid rendering.
    - ``plot_training_curve``: reward/step curve after training.
"""

from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend when a display is unavailable
matplotlib.use("Agg")


class GridWorldVisualizer:
    """Render the grid world as a colour-coded matplotlib figure.

    Colours:
        - White  : empty cell
        - Blue   : agent position
        - Green  : goal position
        - Red    : obstacle
    """

    # RGB colours for each cell type
    _COLOURS = {
        "empty":    [1.0, 1.0, 1.0],
        "agent":    [0.2, 0.4, 1.0],
        "goal":     [0.2, 0.8, 0.2],
        "obstacle": [0.9, 0.2, 0.2],
    }

    def __init__(self, grid_size: int) -> None:
        """
        Initialize the visualizer.

        Args:
            grid_size: Side length of the square grid.
        """
        self.grid_size = grid_size
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def render(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        obstacles: List[Tuple[int, int]],
        episode: int = 0,
        step: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        """Draw the current grid state.

        Args:
            agent_pos:  (row, col) of the agent.
            goal_pos:   (row, col) of the goal.
            obstacles:  List of (row, col) obstacle positions.
            episode:    Current episode index (for title).
            step:       Current step index (for title).
            save_path:  If provided, save the figure to this path instead of
                        displaying it.
        """
        grid = np.ones((self.grid_size, self.grid_size, 3))

        # Paint obstacles
        for r, c in obstacles:
            grid[r, c] = self._COLOURS["obstacle"]

        # Paint goal
        gr, gc = goal_pos
        grid[gr, gc] = self._COLOURS["goal"]

        # Paint agent (overwrites goal colour if coincident — episode ended)
        ar, ac = agent_pos
        grid[ar, ac] = self._COLOURS["agent"]

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        assert self.ax is not None
        self.ax.clear()
        self.ax.imshow(grid, origin="upper")
        self.ax.set_title(f"Episode {episode} | Step {step}")
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True, color="grey", linewidth=0.5)

        if save_path:
            self.fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.pause(0.05)

    def close(self) -> None:
        """Close the matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def plot_training_curve(
    rewards: List[float],
    window: int = 50,
    title: str = "Training Reward Curve",
    save_path: Optional[str] = None,
) -> None:
    """Plot episode rewards and a rolling average.

    Args:
        rewards:   List of per-episode total rewards.
        window:    Rolling-average window size.
        title:     Figure title.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    episodes = list(range(1, len(rewards) + 1))

    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")

    # Rolling average
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window, len(rewards) + 1),
            rolling,
            color="navy",
            linewidth=2,
            label=f"Rolling avg ({window} eps)",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
