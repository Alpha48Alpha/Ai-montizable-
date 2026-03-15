"""
logger.py — Alpha48Alpha AI Lab
=================================
Lightweight training logger that tracks per-episode metrics and prints
formatted progress summaries to stdout.

Designed to be modular so it can be extended (e.g. writing to CSV, TensorBoard,
or W&B) without touching training code.
"""

from __future__ import annotations

import statistics
from typing import Optional


class Logger:
    """
    Records and displays training statistics for a reinforcement learning run.

    Parameters
    ----------
    print_every : int
        Print a summary line every *print_every* episodes.
    window : int
        Number of recent episodes used to compute rolling averages.
    """

    def __init__(self, print_every: int = 10, window: int = 50) -> None:
        self.print_every = print_every
        self.window = window

        # Full history — kept for analysis / plotting after training
        self.episode_rewards: list[float] = []
        self.episode_losses: list[float] = []
        self.episode_lengths: list[int] = []

    # ------------------------------------------------------------------
    # Logging interface
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        loss: float,
        length: int,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Record metrics for a completed episode and optionally print a summary.

        Parameters
        ----------
        episode : int
            1-based episode number.
        total_reward : float
            Sum of undiscounted rewards collected during the episode.
        loss : float
            Policy-gradient loss returned by the agent's ``update()`` call.
        length : int
            Number of steps taken in the episode.
        extra : dict, optional
            Any additional key-value pairs to display in the summary line.
        """
        self.episode_rewards.append(total_reward)
        self.episode_losses.append(loss)
        self.episode_lengths.append(length)

        if episode % self.print_every == 0:
            self._print_summary(episode, extra or {})

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def _print_summary(self, episode: int, extra: dict) -> None:
        """Print a one-line rolling-average summary to stdout."""
        recent_rewards = self.episode_rewards[-self.window :]
        recent_losses = self.episode_losses[-self.window :]
        recent_lengths = self.episode_lengths[-self.window :]

        avg_reward = statistics.mean(recent_rewards)
        avg_loss = statistics.mean(recent_losses)
        avg_length = statistics.mean(recent_lengths)

        # Build optional extra fields string
        extra_str = ""
        if extra:
            extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extra.items())

        print(
            f"[Ep {episode:>5}]  "
            f"avg_reward={avg_reward:+.3f}  "
            f"avg_loss={avg_loss:.4f}  "
            f"avg_steps={avg_length:.1f}"
            f"{extra_str}"
        )

    def summary(self) -> dict:
        """
        Return a dictionary of overall training statistics.

        Useful for end-of-run reporting or programmatic inspection.

        Returns
        -------
        dict
            Keys: ``total_episodes``, ``best_reward``, ``final_avg_reward``,
            ``final_avg_loss``.
        """
        if not self.episode_rewards:
            return {}

        recent = self.episode_rewards[-self.window :]
        return {
            "total_episodes": len(self.episode_rewards),
            "best_reward": max(self.episode_rewards),
            "final_avg_reward": statistics.mean(recent),
            "final_avg_loss": statistics.mean(self.episode_losses[-self.window :]),
        }
