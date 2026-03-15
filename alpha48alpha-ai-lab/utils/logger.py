"""
Training Logger
===============
Lightweight logger for reinforcement learning experiments.

Tracks per-episode metrics (reward, loss, episode length) and
provides formatted console output and optional CSV export.
"""

import csv
import os
import time
from typing import Optional


class Logger:
    """Records and displays training metrics for RL experiments."""

    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        """
        Parameters
        ----------
        log_dir         : Directory where CSV logs are saved.
        experiment_name : Used as the CSV filename prefix.
        """
        self.experiment_name = experiment_name
        self._episode_records: list[dict] = []
        self._start_time = time.time()

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, f"{experiment_name}.csv")
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer: Optional[csv.DictWriter] = None  # initialised on first log

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        loss: float,
        episode_length: int,
    ) -> None:
        """Record metrics for a completed episode and print a summary line.

        Parameters
        ----------
        episode        : Episode index (0-based).
        total_reward   : Cumulative undiscounted reward for the episode.
        loss           : Policy gradient loss returned by the agent update.
        episode_length : Number of steps taken in the episode.
        """
        elapsed = time.time() - self._start_time
        record = {
            "episode": episode,
            "total_reward": round(total_reward, 4),
            "loss": round(loss, 6),
            "episode_length": episode_length,
            "elapsed_s": round(elapsed, 2),
        }
        self._episode_records.append(record)

        # Initialize CSV writer on first call (infers fieldnames from record keys)
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(record.keys())
            )
            self._csv_writer.writeheader()

        self._csv_writer.writerow(record)
        self._csv_file.flush()

        # Console output
        print(
            f"  Ep {episode:>5}  |  reward {total_reward:>8.3f}  |"
            f"  loss {loss:>10.5f}  |  steps {episode_length:>4}  |"
            f"  {elapsed:>7.1f}s"
        )

    def print_header(self) -> None:
        """Print a formatted header row before training begins."""
        print(f"\n{'─' * 66}")
        print(
            f"  {'Ep':>5}  |  {'Reward':>8}  |  {'Loss':>10}  |"
            f"  {'Steps':>4}  |  {'Time':>7}"
        )
        print(f"{'─' * 66}")

    def print_summary(self, last_n: int = 100) -> None:
        """Print a summary of the last *last_n* episodes."""
        records = self._episode_records[-last_n:]
        if not records:
            print("  No episodes logged yet.")
            return

        rewards = [r["total_reward"] for r in records]
        losses = [r["loss"] for r in records]
        print(f"\n{'─' * 66}")
        print(f"  Summary (last {len(records)} episodes):")
        print(f"    Avg reward : {sum(rewards)/len(rewards):.4f}")
        print(f"    Max reward : {max(rewards):.4f}")
        print(f"    Min reward : {min(rewards):.4f}")
        print(f"    Avg loss   : {sum(losses)/len(losses):.6f}")
        print(f"{'─' * 66}\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the CSV file."""
        if not self._csv_file.closed:
            self._csv_file.close()

    def __del__(self) -> None:
        self.close()
