"""Tests for RL Lab.

Covers:
  - Environment API (GridWorld, ContinuousWorld)
  - Agent forward pass and update step (REINFORCE, DQN)
  - Neural network models (MLP, ValueMLP, CNNEncoder)
  - Replay buffer
  - Checkpointing (save + load round-trip)
  - MetricsLogger
  - Experiment (short training run)
  - Visualizer (runs without error)
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Environment tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGridWorld:
    def setup_method(self):
        from rl_lab.envs.grid_world import GridWorld
        self.env = GridWorld(size=5, obstacle_density=0.1, max_steps=50)

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (self.env.obs_dim,)
        assert isinstance(info, dict)

    def test_step_valid_action(self):
        self.env.reset(seed=1)
        for action in range(4):
            obs, reward, terminated, truncated, info = self.env.step(action)
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

    def test_invalid_action_raises(self):
        self.env.reset(seed=2)
        with pytest.raises(ValueError):
            self.env.step(99)

    def test_truncation_on_max_steps(self):
        self.env.reset(seed=3)
        for _ in range(self.env.max_steps - 1):
            _, _, terminated, truncated, _ = self.env.step(0)
            if terminated:
                break
        # final step should truncate (unless accidentally terminated)
        obs, reward, terminated, truncated, _ = self.env.step(0)
        assert terminated or truncated

    def test_render_returns_string(self):
        self.env.reset(seed=4)
        result = self.env.render()
        assert isinstance(result, str)
        assert "A" in result or "S" in result

    def test_obs_dim_matches_shape(self):
        assert self.env.obs_dim == self.env.observation_space["shape"][0]

    def test_is_discrete(self):
        assert self.env.is_discrete is True

    def test_act_dim(self):
        assert self.env.act_dim == 4


class TestContinuousWorld:
    def setup_method(self):
        from rl_lab.envs.continuous_world import ContinuousWorld
        self.env = ContinuousWorld(max_steps=50, speed=0.1, goal_radius=0.05)

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset(seed=0)
        assert obs.shape == (5,)

    def test_step_action(self):
        self.env.reset(seed=1)
        action = np.array([0.5, -0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert obs.shape == (5,)
        assert isinstance(reward, float)

    def test_render_returns_string(self):
        self.env.reset(seed=2)
        result = self.env.render()
        assert isinstance(result, str)

    def test_is_continuous(self):
        assert self.env.is_discrete is False

    def test_act_dim(self):
        assert self.env.act_dim == 2

    def test_goal_reached(self):
        """Force agent onto goal and verify termination."""
        from rl_lab.envs.continuous_world import ContinuousWorld
        env = ContinuousWorld(max_steps=1000, speed=0.5, goal_radius=0.2)
        obs, _ = env.reset(seed=7)
        # Move toward goal repeatedly
        for _ in range(100):
            goal = env._goal
            agent = env._agent
            direction = goal - agent
            action = direction / (np.linalg.norm(direction) + 1e-8)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated:
                break
        # Should have terminated (found goal) within 100 steps with large speed
        assert terminated


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMLP:
    def test_forward_shape(self):
        from rl_lab.models.mlp import MLP
        net = MLP(in_dim=10, out_dim=4, hidden_sizes=[32, 32])
        x = torch.randn(8, 10)
        out = net(x)
        assert out.shape == (8, 4)

    def test_value_mlp_scalar_output(self):
        from rl_lab.models.mlp import ValueMLP
        net = ValueMLP(in_dim=10, hidden_sizes=[32])
        x = torch.randn(5, 10)
        out = net(x)
        assert out.shape == (5, 1)

    def test_policy_mlp(self):
        from rl_lab.models.mlp import PolicyMLP
        net = PolicyMLP(in_dim=8, out_dim=3, hidden_sizes=[16])
        x = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 3)


class TestCNNEncoder:
    def test_forward_shape(self):
        from rl_lab.models.cnn import CNNEncoder
        enc = CNNEncoder(in_channels=3, img_size=32, out_dim=64)
        x = torch.randn(2, 3, 32, 32)
        out = enc(x)
        assert out.shape == (2, 64)

    def test_hwc_input(self):
        from rl_lab.models.cnn import CNNEncoder
        enc = CNNEncoder(in_channels=1, img_size=16, out_dim=32)
        x = torch.randn(3, 16, 16, 1)  # HWC layout
        out = enc(x)
        assert out.shape == (3, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_push_and_sample(self):
        from rl_lab.utils.replay_buffer import ReplayBuffer, Transition
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(Transition(
                obs=np.zeros(4, dtype=np.float32),
                action=i % 2,
                reward=float(i),
                next_obs=np.ones(4, dtype=np.float32),
                done=False,
            ))
        assert len(buf) == 50
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_capacity_limit(self):
        from rl_lab.utils.replay_buffer import ReplayBuffer, Transition
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(Transition(
                obs=np.zeros(2, dtype=np.float32),
                action=0,
                reward=0.0,
                next_obs=np.zeros(2, dtype=np.float32),
                done=False,
            ))
        assert len(buf) == 10  # capped at capacity

    def test_repr(self):
        from rl_lab.utils.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        assert "100" in repr(buf)


# ─────────────────────────────────────────────────────────────────────────────
# Agent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestREINFORCEAgent:
    def setup_method(self):
        from rl_lab.agents.reinforce import REINFORCEAgent
        self.agent = REINFORCEAgent(
            obs_dim=10, act_dim=4, is_discrete=True,
            hidden_sizes=[32], actor_lr=1e-3, critic_lr=1e-3,
        )

    def test_select_action_discrete(self):
        obs = np.zeros(10, dtype=np.float32)
        action = self.agent.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_full_episode_update(self):
        from rl_lab.envs.grid_world import GridWorld
        from rl_lab.agents.reinforce import REINFORCEAgent
        env = GridWorld(size=4, max_steps=20)
        agent = REINFORCEAgent(
            obs_dim=env.obs_dim, act_dim=env.act_dim, is_discrete=True,
            hidden_sizes=[32], actor_lr=1e-3, critic_lr=1e-3,
        )
        obs, _ = env.reset(seed=0)
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            done = terminated or truncated
        metrics = agent.update()
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics

    def test_state_dict_round_trip(self):
        from rl_lab.agents.reinforce import REINFORCEAgent
        agent = REINFORCEAgent(
            obs_dim=10, act_dim=4, is_discrete=True,
            hidden_sizes=[32], actor_lr=1e-3, critic_lr=1e-3,
        )
        # Provide multiple steps so std() doesn't fail on 1 element
        for _ in range(5):
            obs = np.ones(10, dtype=np.float32)
            agent.select_action(obs)
            agent.store_reward(1.0)
        agent.update()
        sd = agent.state_dict()
        agent.load_state_dict(sd)

    def test_continuous_action(self):
        from rl_lab.agents.reinforce import REINFORCEAgent
        agent = REINFORCEAgent(
            obs_dim=5, act_dim=2, is_discrete=False,
            hidden_sizes=[16],
        )
        obs = np.zeros(5, dtype=np.float32)
        action = agent.select_action(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)


class TestDQNAgent:
    def setup_method(self):
        from rl_lab.agents.dqn import DQNAgent
        self.agent = DQNAgent(
            obs_dim=8, act_dim=4, hidden_sizes=[32],
            buffer_capacity=200, batch_size=16, eps_decay_steps=50,
        )

    def test_select_action_returns_int(self):
        obs = np.zeros(8, dtype=np.float32)
        action = self.agent.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_epsilon_decays(self):
        obs = np.zeros(8, dtype=np.float32)
        for _ in range(100):
            self.agent.select_action(obs)
            self.agent.store_transition(obs, 0, 1.0, obs, False)
            self.agent.update()
        assert self.agent.epsilon <= self.agent.eps_start

    def test_update_returns_metrics(self):
        obs = np.zeros(8, dtype=np.float32)
        for _ in range(20):
            self.agent.store_transition(obs, 0, 1.0, obs, False)
        metrics = self.agent.update()
        assert "loss" in metrics

    def test_state_dict_round_trip(self):
        obs = np.zeros(8, dtype=np.float32)
        for _ in range(20):
            self.agent.store_transition(obs, 0, 1.0, obs, False)
        self.agent.update()
        sd = self.agent.state_dict()
        self.agent.load_state_dict(sd)
        assert self.agent._step_count == sd["step_count"]

    def test_target_net_update(self):
        obs = np.zeros(8, dtype=np.float32)
        self.agent.target_update_freq = 5
        for _ in range(100):
            self.agent.store_transition(obs, 0, 1.0, obs, False)
        for _ in range(10):
            self.agent.update()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointing:
    def test_save_and_load_round_trip(self):
        from rl_lab.utils.checkpointing import save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ckpt.pt")
            agent_state = {"weights": torch.tensor([1.0, 2.0])}
            save_checkpoint(
                path=path,
                agent_state=agent_state,
                episode=10,
                total_steps=500,
                best_eval_return=0.75,
                config={"agent": "dqn"},
            )
            assert os.path.exists(path)
            ckpt = load_checkpoint(path)
            assert ckpt["episode"] == 10
            assert ckpt["total_steps"] == 500
            assert abs(ckpt["best_eval_return"] - 0.75) < 1e-6
            assert ckpt["config"]["agent"] == "dqn"
            assert torch.allclose(ckpt["agent_state"]["weights"],
                                  torch.tensor([1.0, 2.0]))

    def test_latest_symlink_created(self):
        from rl_lab.utils.checkpointing import save_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint_ep000010.pt")
            save_checkpoint(
                path=path,
                agent_state={},
                episode=10,
                total_steps=100,
                best_eval_return=0.0,
                config={},
            )
            latest = os.path.join(tmpdir, "checkpoint_latest.pt")
            assert os.path.islink(latest) or os.path.exists(latest)


# ─────────────────────────────────────────────────────────────────────────────
# MetricsLogger tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsLogger:
    def test_log_and_mean(self):
        from rl_lab.utils.metrics import MetricsLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "metrics.jsonl")
            logger = MetricsLogger(log_path=log_path, window=10)
            for i in range(5):
                logger.log({"episode": i, "ep_return": float(i)})
            assert abs(logger.mean("ep_return") - 2.0) < 1e-6
            assert logger.latest("ep_return") == 4.0
            logger.close()

    def test_log_file_written(self):
        from rl_lab.utils.metrics import MetricsLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "metrics.jsonl")
            logger = MetricsLogger(log_path=log_path)
            logger.log({"episode": 1, "ep_return": -0.5})
            logger.close()
            with open(log_path) as fh:
                lines = fh.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["ep_return"] == -0.5

    def test_summary_non_empty(self):
        from rl_lab.utils.metrics import MetricsLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(os.path.join(tmpdir, "m.jsonl"))
            logger.log({"ep_return": 1.0, "loss": 0.5})
            summary = logger.summary()
            assert "ep_return" in summary
            logger.close()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment integration test (very short run)
# ─────────────────────────────────────────────────────────────────────────────

class TestExperiment:
    def _make_config(self, agent: str, env: str, tmpdir: str) -> dict:
        return {
            "experiment_name": f"test_{agent}_{env}",
            "agent": agent,
            "env": env,
            "env_kwargs": {"size": 4, "max_steps": 20} if env == "gridworld"
                          else {"max_steps": 20},
            "agent_kwargs": {"hidden_sizes": [16]}
                            if agent == "dqn"
                            else {"hidden_sizes": [16], "actor_lr": 1e-3, "critic_lr": 1e-3},
            "training": {
                "n_episodes": 5,
                "eval_every": 5,
                "eval_episodes": 2,
                "checkpoint_every": 5,
                "log_every": 5,
                "device": "cpu",
                "seed": 0,
            },
            "output_dir": os.path.join(tmpdir, f"{agent}_{env}"),
        }

    def test_dqn_gridworld_short_run(self):
        from rl_lab.experiment import Experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config("dqn", "gridworld", tmpdir)
            exp = Experiment(config)
            results = exp.run()
            assert "episode_returns" in results
            assert len(results["episode_returns"]) == 5
            assert results["total_steps"] > 0

    def test_reinforce_gridworld_short_run(self):
        from rl_lab.experiment import Experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config("reinforce", "gridworld", tmpdir)
            exp = Experiment(config)
            results = exp.run()
            assert "episode_returns" in results

    def test_reinforce_continuous_short_run(self):
        from rl_lab.experiment import Experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "experiment_name": "test_reinforce_continuous",
                "agent": "reinforce",
                "env": "continuous",
                "env_kwargs": {"max_steps": 20},
                "agent_kwargs": {"hidden_sizes": [16], "actor_lr": 1e-3, "critic_lr": 1e-3},
                "training": {
                    "n_episodes": 5,
                    "eval_every": 5,
                    "eval_episodes": 2,
                    "checkpoint_every": 5,
                    "log_every": 5,
                    "device": "cpu",
                    "seed": 0,
                },
                "output_dir": os.path.join(tmpdir, "reinforce_continuous"),
            }
            exp = Experiment(config)
            results = exp.run()
            assert "episode_returns" in results

    def test_from_config_file(self):
        from rl_lab.experiment import Experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config("dqn", "gridworld", tmpdir)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as fh:
                json.dump(config, fh)
            exp = Experiment.from_config_file(config_path)
            assert exp.name == config["experiment_name"]

    def test_checkpoint_and_resume(self):
        from rl_lab.experiment import Experiment
        from rl_lab.utils.checkpointing import save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config("dqn", "gridworld", tmpdir)
            exp = Experiment(config)
            exp.run()
            # Find latest checkpoint
            run_dir = config["output_dir"]
            latest = os.path.join(run_dir, "checkpoint_latest.pt")
            if os.path.exists(latest):
                exp2 = Experiment(config)
                exp2.resume(latest)
                assert exp2.total_steps >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualize:
    def _write_sample_metrics(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        records = []
        for i in range(1, 31):
            records.append({"episode": i, "ep_return": -1.0 + i * 0.03,
                             "ep_length": 100, "loss": 0.5 - i * 0.01,
                             "epsilon": max(0.05, 1.0 - i * 0.03)})
            if i % 10 == 0:
                records.append({"episode": i, "eval_return": -0.5 + i * 0.02,
                                 "eval_episodes": 5})
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    def test_visualize_generates_plots(self):
        from rl_lab.visualize import visualize
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.jsonl")
            self._write_sample_metrics(metrics_path)
            result = visualize(metrics_path, output_dir=tmpdir, show=False)
            assert os.path.exists(result["learning_curve"])
            assert os.path.exists(result["eval_curve"])

    def test_visualize_no_eval_data(self):
        from rl_lab.visualize import visualize
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.jsonl")
            # Only training records, no eval
            with open(metrics_path, "w") as fh:
                for i in range(5):
                    fh.write(json.dumps({"episode": i, "ep_return": float(i)}) + "\n")
            result = visualize(metrics_path, output_dir=tmpdir, show=False)
            assert result["learning_curve"]  # learning curve should still be generated
            assert result["eval_curve"] == ""  # no eval data
