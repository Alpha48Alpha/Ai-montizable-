"""Visualization module.

Produces publication-quality plots from training metrics:

  1. **Learning curve** — episode return over time with smoothed trend
  2. **Evaluation curve** — periodic evaluation returns
  3. **Training metrics** — loss, epsilon, buffer size (agent-specific)
  4. **Environment snapshot** — ASCII grid (printed) + matplotlib pixel heatmap

All plots are saved to ``<output_dir>/plots/`` as PNG files.

CLI usage
---------
::

    python -m rl_lab.visualize --metrics runs/dqn_gridworld/metrics.jsonl
    python -m rl_lab.visualize --metrics runs/dqn_gridworld/metrics.jsonl --show

Extension hooks
---------------
- Add a ``plot_value_map()`` function to visualise the learned Q-values on
  the GridWorld grid.
- Add a ``plot_trajectory()`` for the ContinuousWorld environment.
- Export plots as SVG for papers / slides.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np


# ─── Helpers ────────────────────────────────────────────────────────────────

def _load_metrics(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _smooth(values: list[float], window: int = 20) -> list[float]:
    if window <= 1 or len(values) < 2:
        return values
    result: list[float] = []
    for i, v in enumerate(values):
        lo = max(0, i - window + 1)
        result.append(float(np.mean(values[lo : i + 1])))
    return result


# ─── Plot functions ─────────────────────────────────────────────────────────

def plot_learning_curve(
    records: list[dict[str, Any]],
    output_dir: str,
    show: bool = False,
) -> str:
    """Plot episode return over training episodes."""
    training = [r for r in records if "ep_return" in r]
    if not training:
        print("No episode return data found.")
        return ""

    episodes = [r["episode"] for r in training]
    returns = [r["ep_return"] for r in training]
    smoothed = _smooth(returns, window=20)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, returns, alpha=0.3, color="steelblue", linewidth=0.8,
            label="Episode return")
    ax.plot(episodes, smoothed, color="steelblue", linewidth=2.0,
            label="Smoothed (window=20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    path = os.path.join(output_dir, "plots", "learning_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_eval_curve(
    records: list[dict[str, Any]],
    output_dir: str,
    show: bool = False,
) -> str:
    """Plot periodic evaluation returns."""
    evals = [r for r in records if "eval_return" in r]
    if not evals:
        print("No evaluation return data found.")
        return ""

    episodes = [r["episode"] for r in evals]
    eval_returns = [r["eval_return"] for r in evals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, eval_returns, marker="o", color="coral", linewidth=2.0,
            label="Eval mean return")
    ax.axhline(max(eval_returns), color="coral", linestyle="--", alpha=0.5,
               label=f"Best = {max(eval_returns):.3f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Return")
    ax.set_title("Evaluation Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    path = os.path.join(output_dir, "plots", "eval_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_agent_metrics(
    records: list[dict[str, Any]],
    output_dir: str,
    show: bool = False,
) -> list[str]:
    """Plot agent-specific training metrics (loss, epsilon, etc.)."""
    saved: list[str] = []
    metric_keys = ["loss", "actor_loss", "critic_loss", "entropy", "epsilon"]
    training = [r for r in records if "ep_return" in r]

    for key in metric_keys:
        values = [r[key] for r in training if key in r]
        if not values:
            continue
        episodes = [r["episode"] for r in training if key in r]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(episodes, values, alpha=0.6, linewidth=1.0, color="mediumseagreen")
        ax.plot(episodes, _smooth(values, window=20), linewidth=2.0,
                color="darkgreen", label=f"Smoothed {key}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(key)
        ax.set_title(f"Training Metric: {key}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        path = os.path.join(output_dir, "plots", f"metric_{key}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        saved.append(path)
        print(f"Saved: {path}")

    return saved


def visualize(
    metrics_path: str,
    output_dir: str | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Generate all plots from a metrics JSONL file.

    Parameters
    ----------
    metrics_path : str
        Path to the ``metrics.jsonl`` produced by :class:`MetricsLogger`.
    output_dir : str | None
        Directory for saving plots.  Defaults to the directory containing
        the metrics file.
    show : bool
        Open matplotlib windows interactively.

    Returns
    -------
    dict with ``"learning_curve"``, ``"eval_curve"``, ``"agent_metrics"``
    """
    if output_dir is None:
        output_dir = os.path.dirname(metrics_path) or "."

    records = _load_metrics(metrics_path)
    print(f"Loaded {len(records)} log entries from {metrics_path}")

    lc = plot_learning_curve(records, output_dir, show=show)
    ec = plot_eval_curve(records, output_dir, show=show)
    am = plot_agent_metrics(records, output_dir, show=show)

    return {"learning_curve": lc, "eval_curve": ec, "agent_metrics": am}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RL Lab training metrics")
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output-dir", default=None, help="Plot output directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()
    visualize(args.metrics, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
