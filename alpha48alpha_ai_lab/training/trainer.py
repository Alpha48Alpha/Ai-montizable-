"""
training/trainer.py — Generic training loop for Alpha48Alpha AI Lab.

The Trainer orchestrates the interaction between an agent and an
environment for a fixed number of episodes.  It is intentionally
agent-agnostic: both PolicyGradientAgent and DQNAgent implement the
same minimal interface that the Trainer relies on.

Agent interface expected by Trainer
-------------------------------------
  * agent.select_action(state) → int
  * agent.store_reward(reward)          [policy-gradient only — no-op for DQN]
  * agent.store_transition(s, a, r, s', done)  [DQN only — no-op for PG]
  * agent.update() → float              [loss or 0.0]
  * agent.end_episode()                 [ε decay for DQN, no-op for PG]
  * agent.save(path)

The Trainer inspects the agent type at construction time to determine
which storage method to call, keeping both agent classes clean.
"""

from __future__ import annotations

import os
import time
from typing import List

from utils.logger import Logger


class Trainer:
    """
    Episode-based training loop.

    Parameters
    ----------
    agent : PolicyGradientAgent | DQNAgent
        The RL agent to train.
    env : GridWorld | WorldSimulator
        The environment to train in.
    logger : Logger
        Metrics logger instance.
    num_episodes : int
        Total number of training episodes.
    max_steps : int
        Maximum steps per episode.
    log_interval : int
        Print metrics every N episodes.
    save_interval : int
        Save a model checkpoint every N episodes.
    checkpoint_dir : str
        Directory for checkpoint files.
    render : bool
        Whether to call env.render() every step.
    """

    def __init__(
        self,
        agent,
        env,
        logger:         Logger,
        num_episodes:   int  = 300,
        max_steps:      int  = 200,
        log_interval:   int  = 10,
        save_interval:  int  = 50,
        checkpoint_dir: str  = "checkpoints",
        render:         bool = False,
    ):
        self.agent          = agent
        self.env            = env
        self.logger         = logger
        self.num_episodes   = num_episodes
        self.max_steps      = max_steps
        self.log_interval   = log_interval
        self.save_interval  = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.render         = render

        # Detect agent type by duck-typing to avoid circular imports.
        # PolicyGradientAgent exposes store_reward(); DQNAgent exposes store_transition().
        self._is_pg  = hasattr(agent, "store_reward")
        self._is_dqn = hasattr(agent, "store_transition")

    # ------------------------------------------------------------------
    # Main training entry-point
    # ------------------------------------------------------------------

    def train(self) -> List[float]:
        """
        Run the full training loop.

        Returns
        -------
        List[float]
            Episode total rewards for all episodes.
        """
        episode_rewards: List[float] = []
        start_time = time.time()

        for episode in range(1, self.num_episodes + 1):
            total_reward, steps, loss = self._run_episode()

            episode_rewards.append(total_reward)
            self.logger.log_episode(
                episode=episode,
                reward=total_reward,
                steps=steps,
                loss=loss,
                epsilon=getattr(self.agent, "epsilon", None),
            )

            # Console logging
            if episode % self.log_interval == 0:
                avg_reward = sum(episode_rewards[-self.log_interval:]) / self.log_interval
                eps_str    = (
                    f"Epsilon: {self.agent.epsilon:.4f}"
                    if self._is_dqn
                    else ""
                )
                elapsed    = time.time() - start_time
                print(
                    f"Episode {episode:5d}/{self.num_episodes} | "
                    f"Avg Reward (last {self.log_interval}): {avg_reward:8.2f} | "
                    f"Steps: {steps:4d} | "
                    f"Loss: {loss:.4f} | "
                    f"{eps_str} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            # Checkpoint
            if episode % self.save_interval == 0:
                self._save_checkpoint(episode)

        # Final checkpoint
        self._save_checkpoint(self.num_episodes)
        self.logger.save()
        print("\nTraining complete.")
        return episode_rewards

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_episode(self):
        """
        Collect a full episode and update the agent.

        Returns
        -------
        total_reward : float
        steps : int
        loss : float
        """
        state        = self.env.reset()
        total_reward = 0.0
        loss         = 0.0

        for step in range(self.max_steps):
            if self.render:
                self.env.render()

            action     = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            # Agent-specific storage
            if self._is_pg:
                self.agent.store_reward(reward)
            elif self._is_dqn:
                self.agent.store_transition(state, action, reward, next_state, done)
                loss = self.agent.update()   # DQN updates every step

            state = next_state

            if done:
                break

        # End-of-episode updates
        if self._is_pg:
            loss = self.agent.update()       # PG updates once per episode
        elif self._is_dqn:
            self.agent.end_episode()         # ε decay + target network sync

        return total_reward, step + 1, loss

    def _save_checkpoint(self, episode: int) -> None:
        """Save agent weights to the checkpoint directory."""
        agent_name = type(self.agent).__name__
        path = os.path.join(self.checkpoint_dir, f"{agent_name}_ep{episode}.pt")
        self.agent.save(path)
