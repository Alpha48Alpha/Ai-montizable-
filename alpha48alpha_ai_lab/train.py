"""
train.py — Entry-point script for Alpha48Alpha AI Lab.

Usage
-----
Train with DQN (default):
    python train.py

Train with policy gradient:
    python train.py --agent pg

Options
-------
  --agent      Agent type: "pg" or "dqn" (default: dqn)
  --episodes   Number of training episodes (default: from config.py)
  --grid-size  Grid world side length (default: 8)
  --obstacles  Number of obstacles (default: 5)
  --seed       Random seed (default: 42)
  --csv        Path to save metrics CSV (optional)
  --curve      Path to save training curve PNG (optional)
  --render-interval  Render grid every N episodes, 0 = never (default: 0)
"""

import argparse

from alpha48alpha_ai_lab.config import NUM_EPISODES, RENDER_INTERVAL
from alpha48alpha_ai_lab.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Alpha48Alpha AI Lab — RL Training Script"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="dqn",
        choices=["pg", "dqn"],
        help="Agent type: 'pg' (policy gradient) or 'dqn' (default: dqn)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help=f"Number of training episodes (default: {NUM_EPISODES})",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Grid world side length (default: 8)",
    )
    parser.add_argument(
        "--obstacles",
        type=int,
        default=5,
        help="Number of obstacles in the grid (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to save training metrics CSV (optional)",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default=None,
        help="Path to save training reward curve PNG (optional)",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=RENDER_INTERVAL,
        help=f"Render grid every N episodes; 0 = never (default: {RENDER_INTERVAL})",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and run training."""
    args = parse_args()

    trainer = Trainer(
        agent_type=args.agent,
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        num_obstacles=args.obstacles,
        seed=args.seed,
        render_interval=args.render_interval,
        csv_path=args.csv,
        curve_path=args.curve,
    )
    trainer.train()


if __name__ == "__main__":
    main()
