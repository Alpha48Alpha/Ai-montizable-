"""Random baseline agent."""

import random

from ai_lab.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random — useful as a baseline."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def select_action(self, state: int) -> int:  # noqa: ARG002
        return random.randrange(self.n_actions)

    def update(self, state, action, reward, next_state, done) -> None:  # noqa: ARG002
        pass  # random agent does not learn
