"""Experiment runner.

``Experiment`` orchestrates the full training lifecycle:
  1. Build environment and agent from a config dict
  2. Run training episodes (REINFORCE on-policy or DQN off-policy)
  3. Periodically evaluate, checkpoint, and log metrics
  4. Save the final trained agent

Usage
-----
::

    from rl_lab.experiment import Experiment
    exp = Experiment.from_config_file("rl_lab/configs/dqn_gridworld.json")
    results = exp.run()

Extension hooks
---------------
- Override ``_build_env()`` / ``_build_agent()`` to plug in custom
  environments or agents without changing this file.
- Add curriculum scheduling by wrapping ``_build_env()`` with a
  ``CurriculumWrapper`` that increases difficulty over time.
- Add a world model by inserting imaginary rollouts between real ones.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import numpy as np
import torch

from rl_lab.agents.dqn import DQNAgent
from rl_lab.agents.reinforce import REINFORCEAgent
from rl_lab.envs.grid_world import GridWorld
from rl_lab.envs.continuous_world import ContinuousWorld
from rl_lab.utils.checkpointing import save_checkpoint, load_checkpoint
from rl_lab.utils.metrics import MetricsLogger


class Experiment:
    """Full training experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration (mirrors the JSON config files).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.name = config.get("experiment_name", "experiment")
        training = config.get("training", {})
        self.n_episodes: int = training.get("n_episodes", 500)
        self.eval_every: int = training.get("eval_every", 50)
        self.eval_episodes: int = training.get("eval_episodes", 10)
        self.checkpoint_every: int = training.get("checkpoint_every", 100)
        self.log_every: int = training.get("log_every", 10)
        self.device: str = training.get("device", "cpu")
        self.seed: int = training.get("seed", 0)
        self.output_dir: str = config.get("output_dir", f"runs/{self.name}")

        self._set_seeds(self.seed)
        self.env = self._build_env()
        self.agent = self._build_agent()
        self.logger = MetricsLogger(
            log_path=os.path.join(self.output_dir, "metrics.jsonl"),
            window=100,
        )
        self.best_eval_return: float = float("-inf")
        self.total_steps: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config_file(cls, path: str) -> "Experiment":
        """Create an Experiment from a JSON config file."""
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
        return cls(config)

    def run(self) -> dict:
        """Run the full training loop.

        Returns
        -------
        dict with keys ``episode_returns``, ``eval_returns``,
                       ``best_eval_return``, ``total_steps``
        """
        print(f"\n{'='*60}")
        print(f"  Experiment : {self.name}")
        print(f"  Agent      : {self.config['agent'].upper()}")
        print(f"  Env        : {self.config['env']}")
        print(f"  Episodes   : {self.n_episodes}")
        print(f"  Output dir : {self.output_dir}")
        print(f"{'='*60}\n")

        episode_returns: list[float] = []
        eval_returns: list[float] = []
        t0 = time.time()

        for ep in range(1, self.n_episodes + 1):
            ep_return, ep_length, agent_metrics = self._run_episode(train=True)
            self.total_steps += ep_length
            episode_returns.append(ep_return)

            metrics = {
                "episode": ep,
                "total_steps": self.total_steps,
                "ep_return": ep_return,
                "ep_length": ep_length,
                **agent_metrics,
            }
            self.logger.log(metrics)

            if ep % self.log_every == 0:
                elapsed = time.time() - t0
                avg_return = sum(episode_returns[-self.log_every:]) / self.log_every
                print(
                    f"Ep {ep:5d}/{self.n_episodes}  "
                    f"avg_return={avg_return:+.3f}  "
                    f"steps={self.total_steps}  "
                    f"elapsed={elapsed:.1f}s"
                )

            if ep % self.eval_every == 0:
                mean_eval = self._evaluate()
                eval_returns.append(mean_eval)
                eval_metrics = {
                    "episode": ep,
                    "eval_return": mean_eval,
                    "eval_episodes": self.eval_episodes,
                }
                self.logger.log(eval_metrics)
                print(f"  ↳ Eval  mean_return={mean_eval:+.3f}")

                if mean_eval > self.best_eval_return:
                    self.best_eval_return = mean_eval
                    best_path = os.path.join(self.output_dir, "checkpoint_best.pt")
                    save_checkpoint(
                        path=best_path,
                        agent_state=self.agent.state_dict(),
                        episode=ep,
                        total_steps=self.total_steps,
                        best_eval_return=self.best_eval_return,
                        config=self.config,
                    )
                    print(f"  ↳ New best eval return: {mean_eval:+.3f} → saved")

            if ep % self.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.output_dir, f"checkpoint_ep{ep:06d}.pt"
                )
                save_checkpoint(
                    path=ckpt_path,
                    agent_state=self.agent.state_dict(),
                    episode=ep,
                    total_steps=self.total_steps,
                    best_eval_return=self.best_eval_return,
                    config=self.config,
                )

        self.logger.close()
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best eval return  : {self.best_eval_return:+.3f}")
        print(f"  Total steps       : {self.total_steps}")
        print(f"  Output dir        : {self.output_dir}")
        print(f"{'='*60}\n")

        return {
            "episode_returns": episode_returns,
            "eval_returns": eval_returns,
            "best_eval_return": self.best_eval_return,
            "total_steps": self.total_steps,
        }

    def resume(self, checkpoint_path: str) -> None:
        """Restore agent weights and counters from a checkpoint."""
        ckpt = load_checkpoint(checkpoint_path)
        self.agent.load_state_dict(ckpt["agent_state"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.best_eval_return = ckpt.get("best_eval_return", float("-inf"))
        print(f"Resumed from {checkpoint_path} "
              f"(ep={ckpt.get('episode', '?')}, steps={self.total_steps})")

    # ------------------------------------------------------------------ #
    #  Training / evaluation helpers                                       #
    # ------------------------------------------------------------------ #

    def _run_episode(self, train: bool = True) -> tuple[float, int, dict]:
        """Run a single episode, optionally training the agent.

        Returns
        -------
        (episode_return, episode_length, agent_metrics_dict)
        """
        obs, _ = self.env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        agent_metrics: dict[str, Any] = {}

        agent_type = self.config["agent"]

        while not done:
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if train:
                if agent_type == "reinforce":
                    self.agent.store_reward(reward)  # type: ignore[attr-defined]
                elif agent_type == "dqn":
                    self.agent.store_transition(obs, action, reward, next_obs, terminated)  # type: ignore[attr-defined]
                    metrics = self.agent.update()
                    if metrics:
                        agent_metrics.update(metrics)

            obs = next_obs
            ep_return += reward
            ep_length += 1

        # REINFORCE: update once per episode (on-policy)
        if train and agent_type == "reinforce":
            agent_metrics = self.agent.update()  # type: ignore[attr-defined]

        return ep_return, ep_length, agent_metrics

    def _evaluate(self) -> float:
        """Run ``eval_episodes`` greedy episodes and return the mean return."""
        self.agent.set_eval()
        returns: list[float] = []
        orig_eps = getattr(self.agent, "eps_end", None)  # DQN: temporarily set eps=0
        if orig_eps is not None:
            self.agent.eps_end = 0.0  # type: ignore[attr-defined]

        for _ in range(self.eval_episodes):
            ep_return, _, _ = self._run_episode(train=False)
            returns.append(ep_return)

        if orig_eps is not None:
            self.agent.eps_end = orig_eps  # type: ignore[attr-defined]
        self.agent.set_train()
        return float(np.mean(returns))

    # ------------------------------------------------------------------ #
    #  Factory methods                                                     #
    # ------------------------------------------------------------------ #

    def _build_env(self):
        """Instantiate the environment from config."""
        env_name = self.config["env"]
        kwargs = self.config.get("env_kwargs", {})
        if env_name == "gridworld":
            return GridWorld(**kwargs)
        if env_name == "continuous":
            return ContinuousWorld(**kwargs)
        raise ValueError(f"Unknown env '{env_name}'. Add it to Experiment._build_env().")

    def _build_agent(self):
        """Instantiate the agent from config."""
        agent_name = self.config["agent"]
        kwargs = self.config.get("agent_kwargs", {})
        obs_dim = self.env.obs_dim
        act_dim = self.env.act_dim

        if agent_name == "dqn":
            return DQNAgent(obs_dim=obs_dim, act_dim=act_dim,
                            device=self.device, **kwargs)
        if agent_name == "reinforce":
            return REINFORCEAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                is_discrete=self.env.is_discrete,
                device=self.device,
                **kwargs,
            )
        raise ValueError(
            f"Unknown agent '{agent_name}'. Add it to Experiment._build_agent()."
        )

    # ------------------------------------------------------------------ #
    #  Seeding                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
