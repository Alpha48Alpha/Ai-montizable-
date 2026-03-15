"""
trainer.py — Unified training loop for Alpha48Alpha AI Lab.

Supports two modes selectable via the ``agent_type`` parameter:
    - ``"pg"`` : REINFORCE policy-gradient agent
    - ``"dqn"``: Deep Q-Network agent

Usage::

    from alpha48alpha_ai_lab.training.trainer import Trainer
    trainer = Trainer(agent_type="dqn")
    trainer.train()
"""

from typing import Optional

from alpha48alpha_ai_lab.agents.dqn_agent import DQNAgent
from alpha48alpha_ai_lab.agents.policy_gradient_agent import PolicyGradientAgent
from alpha48alpha_ai_lab.config import (
    BATCH_SIZE,
    LOG_INTERVAL,
    NUM_EPISODES,
    RENDER_INTERVAL,
    TARGET_UPDATE_FREQ,
)
from alpha48alpha_ai_lab.environments.world_simulator import WorldSimulator
from alpha48alpha_ai_lab.utils.logger import TrainingLogger
from alpha48alpha_ai_lab.utils.visualization import plot_training_curve


class Trainer:
    """Orchestrates the training loop for RL agents in the grid world.

    Args:
        agent_type:       ``"pg"`` for policy gradient, ``"dqn"`` for DQN.
        num_episodes:     Number of training episodes.
        grid_size:        Side length of the grid world.
        num_obstacles:    Number of obstacles in the grid world.
        seed:             RNG seed for reproducibility.
        log_interval:     Print stats every N episodes.
        render_interval:  Render the grid every N episodes (0 = never).
        csv_path:         Optional path to save training metrics CSV.
        curve_path:       Optional path to save the training reward curve PNG.
    """

    def __init__(
        self,
        agent_type: str = "dqn",
        num_episodes: int = NUM_EPISODES,
        grid_size: int = 8,
        num_obstacles: int = 5,
        seed: Optional[int] = 42,
        log_interval: int = LOG_INTERVAL,
        render_interval: int = RENDER_INTERVAL,
        csv_path: Optional[str] = None,
        curve_path: Optional[str] = None,
    ) -> None:
        self.agent_type = agent_type.lower()
        self.num_episodes = num_episodes
        self.render_interval = render_interval
        self.curve_path = curve_path

        # Environment
        self.sim = WorldSimulator(grid_size=grid_size, num_obstacles=num_obstacles, seed=seed)

        # Agent
        state_size = self.sim.state_size
        num_actions = self.sim.num_actions

        if self.agent_type == "pg":
            self.agent = PolicyGradientAgent(state_size=state_size, num_actions=num_actions)
        elif self.agent_type == "dqn":
            self.agent = DQNAgent(state_size=state_size, num_actions=num_actions)
        else:
            raise ValueError(f"Unknown agent_type '{agent_type}'. Choose 'pg' or 'dqn'.")

        # Logger
        self.logger = TrainingLogger(log_interval=log_interval, csv_path=csv_path)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        print(f"Starting {self.agent_type.upper()} training for {self.num_episodes} episodes …")

        if self.agent_type == "pg":
            self._train_pg()
        else:
            self._train_dqn()

        self.logger.save()

        if self.curve_path:
            plot_training_curve(
                rewards=self.logger.rewards,
                title=f"Alpha48Alpha AI Lab — {self.agent_type.upper()} Training",
                save_path=self.curve_path,
            )

        print("Training complete.")

    # ------------------------------------------------------------------
    # Agent-specific loops
    # ------------------------------------------------------------------

    def _train_pg(self) -> None:
        """Policy-gradient (REINFORCE) training loop."""
        assert isinstance(self.agent, PolicyGradientAgent)

        for ep in range(self.num_episodes):
            state = self.sim.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.sim.step(action)
                self.agent.store_reward(reward)
                total_reward += reward
                steps += 1
                state = next_state

            self.agent.update()
            self.logger.log_episode(ep, total_reward, steps)

            if self.render_interval > 0 and (ep + 1) % self.render_interval == 0:
                self.sim.render()

    def _train_dqn(self) -> None:
        """DQN training loop with experience replay and target network."""
        assert isinstance(self.agent, DQNAgent)

        for ep in range(self.num_episodes):
            state = self.sim.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.sim.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                total_reward += reward
                steps += 1
                state = next_state

                # Update the online network after each step
                self.agent.update()

            # ε decay and periodic target network sync
            self.agent.decay_epsilon()
            if (ep + 1) % TARGET_UPDATE_FREQ == 0:
                self.agent.sync_target()

            self.logger.log_episode(ep, total_reward, steps)

            if self.render_interval > 0 and (ep + 1) % self.render_interval == 0:
                self.sim.render()
