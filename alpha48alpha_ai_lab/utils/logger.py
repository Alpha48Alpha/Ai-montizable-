"""
logger.py — Lightweight training metrics logger for Alpha48Alpha AI Lab.

Tracks per-episode rewards and step counts, computes rolling averages, and
writes a CSV summary file at the end of training.
"""

import csv
import time
from pathlib import Path
from typing import List, Optional


class TrainingLogger:
    """Logs training metrics to the console and optionally to a CSV file.

    Usage::

        logger = TrainingLogger(log_interval=50, csv_path="logs/run.csv")
        for ep in range(num_episodes):
            logger.log_episode(episode=ep, reward=total_reward, steps=steps)
        logger.save()
    """

    def __init__(
        self,
        log_interval: int = 50,
        csv_path: Optional[str] = None,
        window: int = 100,
    ) -> None:
        """
        Initialize the logger.

        Args:
            log_interval: Print summary every N episodes.
            csv_path:     Optional path to write a CSV metrics file.
            window:       Rolling-average window size.
        """
        self.log_interval = log_interval
        self.csv_path = Path(csv_path) if csv_path else None
        self.window = window

        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.steps_list: List[int] = []
        self._start_time: float = time.time()

    def log_episode(self, episode: int, reward: float, steps: int) -> None:
        """Record metrics for a completed episode and print if at log interval.

        Args:
            episode: Episode index (0-based).
            reward:  Total undiscounted reward for the episode.
            steps:   Number of steps taken in the episode.
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps_list.append(steps)

        if (episode + 1) % self.log_interval == 0:
            avg_reward = self._rolling_avg(self.rewards)
            avg_steps = self._rolling_avg(self.steps_list)
            elapsed = time.time() - self._start_time
            print(
                f"Episode {episode + 1:>6} | "
                f"Avg Reward (last {self.window}): {avg_reward:+.3f} | "
                f"Avg Steps: {avg_steps:.1f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    def save(self) -> None:
        """Write all recorded metrics to the CSV file (if configured)."""
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "steps"])
            writer.writerows(zip(self.episodes, self.rewards, self.steps_list))
        print(f"Metrics saved to {self.csv_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rolling_avg(self, values: list) -> float:
        """Compute the rolling average over the last ``window`` entries."""
        recent = values[-self.window:]
        return sum(recent) / len(recent) if recent else 0.0
