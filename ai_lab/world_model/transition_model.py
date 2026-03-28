"""Frequency-based tabular transition model."""

from collections import defaultdict
from typing import Dict, List, Tuple

from ai_lab.world_model.base_model import BaseWorldModel


class TransitionModel(BaseWorldModel):
    """Tabular transition model that learns from observed (s, a, s', r) tuples.

    For each (state, action) pair the model records every observed next-state
    and reward.  :meth:`predict` returns the most-frequently seen next state and
    the average reward — a simple but interpretable baseline for model-based RL.

    Parameters
    ----------
    n_states:  Number of discrete states.
    n_actions: Number of discrete actions.
    """

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        # counts[(s, a)][s'] = number of times transition was observed
        self._counts: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # rewards[(s, a)] = list of observed rewards
        self._rewards: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # BaseWorldModel interface
    # ------------------------------------------------------------------

    def observe(self, state: int, action: int, next_state: int, reward: float) -> None:
        """Record a real environment transition."""
        self._counts[(state, action)][next_state] += 1
        self._rewards[(state, action)].append(reward)

    def predict(self, state: int, action: int) -> Tuple[int, float]:
        """Return the most likely next state and mean reward.

        If the (state, action) pair has never been observed, the model returns
        *state* unchanged and a reward of 0.0.
        """
        key = (state, action)
        if not self._counts[key]:
            return state, 0.0
        counts = self._counts[key]
        best_next = max(counts, key=lambda s: counts[s])
        mean_reward = sum(self._rewards[key]) / len(self._rewards[key])
        return best_next, mean_reward

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def transition_count(self, state: int, action: int) -> int:
        """Return the total number of observed transitions for ``(state, action)``."""
        return sum(self._counts[(state, action)].values())
