"""Base agent interface.

All RL agents in RL Lab subclass BaseAgent.  The interface mirrors the
standard RL interaction loop so experiment runners can swap agents
without any other code changes.

Extension points
----------------
- Override ``preprocess_obs()`` for image / language observations
- Add ``intrinsic_reward()`` for curiosity / exploration bonuses
- Add ``human_feedback()`` for RLHF / human-in-the-loop research
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch


class BaseAgent(abc.ABC):
    """Abstract base class for all RL Lab agents.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the (flat) observation vector.
    act_dim : int
        Number of discrete actions (or continuous action dimension).
    device : str
        PyTorch device string ("cpu", "cuda", "mps", …).
    """

    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu") -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)

    # ------------------------------------------------------------------ #
    #  Abstract API                                                        #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray) -> Any:
        """Choose an action given the current observation.

        Parameters
        ----------
        obs : np.ndarray  (shape: ``(obs_dim,)``)

        Returns
        -------
        action : int | np.ndarray
        """

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """Perform one learning update.

        Returns
        -------
        metrics : dict
            Dictionary of scalar training metrics (loss, etc.) for logging.
        """

    @abc.abstractmethod
    def state_dict(self) -> dict:
        """Return all agent state needed for checkpointing."""

    @abc.abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore agent state from a checkpoint dictionary."""

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    def preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a numpy observation to a float tensor on the agent's device."""
        return torch.from_numpy(obs.astype(np.float32)).to(self.device)

    def set_train(self) -> None:
        """Switch all PyTorch modules to training mode."""
        for attr in self.__dict__.values():
            if isinstance(attr, torch.nn.Module):
                attr.train()

    def set_eval(self) -> None:
        """Switch all PyTorch modules to evaluation mode."""
        for attr in self.__dict__.values():
            if isinstance(attr, torch.nn.Module):
                attr.eval()
