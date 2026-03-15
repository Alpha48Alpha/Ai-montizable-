#!/usr/bin/env python3
"""Main training entry point for RL Lab.

Usage
-----
::

    # Train DQN on GridWorld (default)
    python train.py --config rl_lab/configs/dqn_gridworld.json

    # Train REINFORCE on GridWorld
    python train.py --config rl_lab/configs/reinforce_gridworld.json

    # Train REINFORCE on the continuous navigation task
    python train.py --config rl_lab/configs/reinforce_continuous.json

    # Resume from a checkpoint
    python train.py --config rl_lab/configs/dqn_gridworld.json \\
                    --resume runs/dqn_gridworld/checkpoint_latest.pt

    # After training, generate plots
    python -m rl_lab.visualize --metrics runs/dqn_gridworld/metrics.jsonl

    # Evaluate the best checkpoint
    python -m rl_lab.evaluate --checkpoint runs/dqn_gridworld/checkpoint_best.pt

Architecture overview
---------------------
  train.py               ← this file (CLI entry point)
  rl_lab/
    experiment.py        ← orchestrates the training loop
    envs/                ← simulated worlds
      grid_world.py      ← 8×8 discrete navigation task
      continuous_world.py← 2-D continuous navigation task
    agents/              ← RL algorithms
      reinforce.py       ← Monte-Carlo policy gradient + baseline
      dqn.py             ← Double DQN with experience replay
    models/              ← neural network architectures
      mlp.py             ← MLP, PolicyMLP, ValueMLP
      cnn.py             ← CNN encoder (for image observations)
    utils/               ← shared utilities
      replay_buffer.py   ← uniform experience replay
      checkpointing.py   ← save / load .pt checkpoints
      metrics.py         ← JSONL metrics logger
    configs/             ← experiment JSON configurations
    evaluate.py          ← deterministic evaluation script
    visualize.py         ← learning-curve & metric plots
"""

import argparse
import sys

from rl_lab.experiment import Experiment
from rl_lab.visualize import visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL Lab — production-grade reinforcement learning research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON experiment config (see rl_lab/configs/)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a .pt checkpoint to resume training from",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating plots after training",
    )
    args = parser.parse_args()

    exp = Experiment.from_config_file(args.config)

    if args.resume:
        exp.resume(args.resume)

    exp.run()

    if not args.no_visualize:
        import os
        metrics_path = os.path.join(exp.output_dir, "metrics.jsonl")
        if os.path.exists(metrics_path):
            print("\nGenerating plots …")
            visualize(metrics_path, output_dir=exp.output_dir)


if __name__ == "__main__":
    main()
