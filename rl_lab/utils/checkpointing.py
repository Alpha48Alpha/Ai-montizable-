"""Checkpointing: save and restore experiment state.

Saves a checkpoint dict containing:
  - agent state_dict (all model and optimiser weights)
  - episode number, total steps, and best evaluation return
  - full experiment config

Checkpoints are stored as ``<run_dir>/checkpoint_ep<episode>.pt`` with
a symlink ``<run_dir>/checkpoint_latest.pt`` always pointing to the most
recent save.

Extension hooks
---------------
- Extend the checkpoint dict to store world-model weights, replay-buffer
  contents, or normalisation statistics.
- Add versioning / migration logic for long-running experiments.
"""

from __future__ import annotations

import os

import torch


def save_checkpoint(
    path: str,
    agent_state: dict,
    episode: int,
    total_steps: int,
    best_eval_return: float,
    config: dict,
) -> None:
    """Save an experiment checkpoint to *path*.

    Parameters
    ----------
    path : str
        Destination ``.pt`` file path.
    agent_state : dict
        Result of ``agent.state_dict()``.
    episode : int
    total_steps : int
    best_eval_return : float
    config : dict
        Full experiment configuration dict.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "agent_state": agent_state,
        "episode": episode,
        "total_steps": total_steps,
        "best_eval_return": best_eval_return,
        "config": config,
    }
    torch.save(payload, path)

    # Keep a "latest" pointer in the same directory
    run_dir = os.path.dirname(path)
    latest = os.path.join(run_dir, "checkpoint_latest.pt")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    # Use a relative symlink so the directory stays portable
    os.symlink(os.path.basename(path), latest)


def load_checkpoint(path: str) -> dict:
    """Load and return a checkpoint dictionary from *path*.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.

    Returns
    -------
    dict with keys: ``agent_state``, ``episode``, ``total_steps``,
                    ``best_eval_return``, ``config``
    """
    return torch.load(path, map_location="cpu", weights_only=False)
