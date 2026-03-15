"""Tabular Q-learning agent."""

import random
from typing import List

from ai_lab.agents.base_agent import BaseAgent


class QAgent(BaseAgent):
    """ε-greedy tabular Q-learning agent.

    Parameters
    ----------
    n_states:      Number of discrete states in the environment.
    n_actions:     Number of discrete actions available.
    learning_rate: Step size α for Q-value updates (default 0.1).
    discount:      Discount factor γ for future rewards (default 0.99).
    epsilon:       Initial exploration probability (default 1.0).
    epsilon_decay: Multiplicative decay applied after each update (default 0.999).
    epsilon_min:   Minimum value for epsilon (default 0.01).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
    ) -> None:
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table: shape (n_states, n_actions), initialised to zero
        self.q_table: List[List[float]] = [
            [0.0] * n_actions for _ in range(n_states)
        ]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self._best_action(state)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Bellman update for Q(state, action)."""
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - current_q)
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_action(self, state: int) -> int:
        row = self.q_table[state]
        best = max(row)
        # Break ties randomly for a more robust policy
        best_actions = [a for a, q in enumerate(row) if q == best]
        return random.choice(best_actions)
