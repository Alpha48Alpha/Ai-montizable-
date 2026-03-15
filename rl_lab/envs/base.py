"""Base environment interface.

All simulated worlds in RL Lab must subclass BaseEnv and implement
the abstract methods below.  This ensures every environment exposes
a consistent, Gym-compatible API so agents and experiment runners
never need environment-specific code.

Extension points
----------------
- Override ``_build_obs()`` to add richer observations (images, language, etc.)
- Subclass with a ``human_step()`` method for human-in-the-loop research
- Add a ``world_model_obs()`` method to expose latent states for world-model agents
"""

from __future__ import annotations

import abc
import numpy as np
from typing import Any


class BaseEnv(abc.ABC):
    """Abstract base class for all RL-lab environments.

    Attributes
    ----------
    observation_space : dict
        ``{"shape": tuple, "dtype": np.dtype}``
    action_space : dict
        ``{"n": int}`` for discrete, ``{"low": ndarray, "high": ndarray}`` for continuous
    """

    # Sub-classes must set these in their ``__init__``.
    observation_space: dict
    action_space: dict

    # ------------------------------------------------------------------ #
    #  Abstract API                                                        #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset to an initial state.

        Returns
        -------
        obs : np.ndarray
            The first observation.
        info : dict
            Auxiliary diagnostic information.
        """

    @abc.abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply *action* and advance the environment by one time-step.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
            Episode ended because of a terminal state (e.g. goal reached).
        truncated : bool
            Episode ended because of a time-limit.
        info : dict
        """

    @abc.abstractmethod
    def render(self) -> str:
        """Return a human-readable string representation of the current state."""

    # ------------------------------------------------------------------ #
    #  Convenience helpers (may be overridden)                            #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Clean up resources.  Override if needed."""

    @property
    def is_discrete(self) -> bool:
        """True when the action space is discrete."""
        return "n" in self.action_space

    @property
    def obs_dim(self) -> int:
        """Flat observation dimensionality."""
        shape = self.observation_space["shape"]
        dim = 1
        for s in shape:
            dim *= s
        return dim

    @property
    def act_dim(self) -> int:
        """Number of discrete actions, or continuous action dimensionality."""
        if self.is_discrete:
            return self.action_space["n"]
        return len(self.action_space["low"])
