"""Metrics logger.

Tracks scalar training and evaluation metrics, writes them to a
newline-delimited JSON log file, and exposes rolling statistics for
the training loop.

Log file format
---------------
Each entry is a JSON object on its own line::

    {"episode": 1, "total_steps": 200, "ep_return": -1.23, "ep_length": 200}
    {"episode": 2, "total_steps": 350, "ep_return": -0.94, "ep_length": 150}

This format is easily ingestible by pandas, Spark, or any log-analysis tool.

Extension hooks
---------------
- Add a TensorBoard writer (``torch.utils.tensorboard``) alongside the
  JSON log for real-time dashboards.
- Attach a remote experiment-tracking backend (MLflow, W&B) by overriding
  ``_flush()``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from typing import Any


class MetricsLogger:
    """Tracks, aggregates, and persists scalar training metrics.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file.  The parent directory is created
        automatically.
    window : int
        Rolling-window size for smoothed statistics.
    """

    def __init__(self, log_path: str, window: int = 100) -> None:
        self.log_path = log_path
        self.window = window
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    # ------------------------------------------------------------------ #
    #  Context-manager support                                             #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def log(self, metrics: dict[str, Any]) -> None:
        """Record a metrics dict.  Writes one JSON line to the log file."""
        self._fh.write(json.dumps(metrics) + "\n")
        self._fh.flush()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._history[k].append(float(v))

    def mean(self, key: str) -> float | None:
        """Rolling mean of *key* over the last ``window`` records."""
        buf = self._history.get(key)
        if not buf:
            return None
        return sum(buf) / len(buf)

    def latest(self, key: str) -> float | None:
        """Most recent value of *key*."""
        buf = self._history.get(key)
        if not buf:
            return None
        return buf[-1]

    def summary(self) -> str:
        """One-line human-readable summary of recent metrics."""
        parts = []
        for k, buf in sorted(self._history.items()):
            if buf:
                parts.append(f"{k}={buf[-1]:.4f}(avg {sum(buf)/len(buf):.4f})")
        return "  ".join(parts)

    def close(self) -> None:
        """Flush and close the log file."""
        self._fh.close()

    def __repr__(self) -> str:
        return f"MetricsLogger(log_path={self.log_path!r})"
