"""
utils/logger.py — Reward and metrics logger for Alpha48Alpha AI Lab.

The Logger records per-episode metrics (reward, steps, loss, epsilon)
to an in-memory list and can serialise them to a CSV file for later
analysis (e.g. with pandas or a spreadsheet).
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional


class Logger:
    """
    Lightweight training metrics logger.

    Parameters
    ----------
    agent_name : str
        Short name of the agent (used in the output filename).
    env_name : str
        Short name of the environment (used in the output filename).
    log_dir : str
        Directory where the CSV log file will be written (default: "logs").
    """

    def __init__(
        self,
        agent_name: str = "agent",
        env_name:   str = "env",
        log_dir:    str = "logs",
    ):
        self.agent_name = agent_name
        self.env_name   = env_name
        self.log_dir    = log_dir

        # In-memory record store
        self._records: List[dict] = []

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode:  int,
        reward:   float,
        steps:    int,
        loss:     float = 0.0,
        epsilon:  Optional[float] = None,
    ) -> None:
        """
        Record metrics for a single episode.

        Parameters
        ----------
        episode : int          — episode number (1-indexed).
        reward  : float        — total undiscounted episode reward.
        steps   : int          — number of steps taken in the episode.
        loss    : float        — agent loss value (0 if not applicable).
        epsilon : float | None — current exploration rate (DQN only).
        """
        record = {
            "episode": episode,
            "reward":  reward,
            "steps":   steps,
            "loss":    loss,
            "epsilon": epsilon if epsilon is not None else "",
        }
        self._records.append(record)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: Optional[str] = None) -> str:
        """
        Write all logged episodes to a CSV file.

        Parameters
        ----------
        filename : str | None
            Path to the output file.  If None, a default name is
            constructed from the agent and environment names.

        Returns
        -------
        str — path to the written CSV file.
        """
        os.makedirs(self.log_dir, exist_ok=True)

        if filename is None:
            filename = os.path.join(
                self.log_dir, f"{self.agent_name}_{self.env_name}_log.csv"
            )

        if not self._records:
            return filename

        fieldnames = ["episode", "reward", "steps", "loss", "epsilon"]
        with open(filename, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)

        print(f"Training log saved to {filename}")
        return filename

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def rewards(self) -> List[float]:
        """Return the list of per-episode rewards."""
        return [r["reward"] for r in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"Logger(agent={self.agent_name!r}, env={self.env_name!r}, "
            f"episodes_logged={len(self._records)})"
        )
