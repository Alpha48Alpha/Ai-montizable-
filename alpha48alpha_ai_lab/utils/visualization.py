"""
utils/visualization.py — Matplotlib visualization helpers.

Provides:
  * plot_rewards()      — plot the episode reward curve with a rolling average.
  * render_grid_world() — render a GridWorld frame as a colour image.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Reward curve
# ---------------------------------------------------------------------------

def plot_rewards(
    rewards:    List[float],
    title:      str = "Episode Rewards",
    window:     int = 20,
    save_path:  Optional[str] = None,
    show:       bool = False,
) -> None:
    """
    Plot episode rewards and a rolling-average smoothed curve.

    Parameters
    ----------
    rewards : List[float]
        Per-episode total rewards in chronological order.
    title : str
        Plot title.
    window : int
        Rolling-average window size (default: 20 episodes).
    save_path : str | None
        If provided, save the figure to this path (.png recommended).
    show : bool
        If True, display the figure interactively (requires a display).
    """
    episodes = np.arange(1, len(rewards) + 1)

    # Compute rolling average
    rolling_avg = _rolling_average(rewards, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards,     alpha=0.4, color="steelblue",   label="Episode reward")
    ax.plot(episodes, rolling_avg, linewidth=2, color="darkorange",
            label=f"Rolling avg (window={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Reward plot saved to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Grid-world frame renderer
# ---------------------------------------------------------------------------

def render_grid_world(
    agent_pos:  tuple,
    goal_pos:   tuple,
    grid_size:  int,
    obstacles:  Optional[List[tuple]] = None,
    save_path:  Optional[str] = None,
    show:       bool = False,
    step:       int = 0,
    reward:     float = 0.0,
) -> None:
    """
    Render a single grid-world frame using matplotlib.

    Parameters
    ----------
    agent_pos : (row, col)   — current agent position.
    goal_pos  : (row, col)   — goal cell position.
    grid_size : int          — side length of the square grid.
    obstacles : list[(row, col)] | None
        Obstacle cell positions (optional).
    save_path : str | None
        If provided, the frame is saved as an image.
    show : bool
        If True, display the frame interactively.
    step : int
        Current step number (displayed in the title).
    reward : float
        Most-recent reward (displayed in the title).
    """
    obstacles = obstacles or []

    # Build an RGB image: white background
    img = np.ones((grid_size, grid_size, 3), dtype=np.float32)

    # Colour obstacle cells dark grey
    for r, c in obstacles:
        img[r, c] = [0.2, 0.2, 0.2]

    # Colour goal cell green
    gr, gc = goal_pos
    img[gr, gc] = [0.2, 0.8, 0.2]

    # Colour agent cell blue
    ar, ac = agent_pos
    img[ar, ac] = [0.2, 0.4, 0.9]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Step {step} | Reward {reward:.2f}")

    # Legend
    legend_elements = [
        mpatches.Patch(color=[0.2, 0.4, 0.9], label="Agent"),
        mpatches.Patch(color=[0.2, 0.8, 0.2], label="Goal"),
        mpatches.Patch(color=[0.2, 0.2, 0.2], label="Obstacle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=100)

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rolling_average(values: List[float], window: int) -> np.ndarray:
    """Compute a causal rolling average (no look-ahead)."""
    result = np.zeros(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start : i + 1])
    return result
