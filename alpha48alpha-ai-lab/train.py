"""
train.py — Alpha48Alpha AI Lab
================================
Main training script for the REINFORCE policy-gradient agent on SimpleWorld.

Run
---
    python train.py [--episodes N] [--grid-size N] [--lr F] [--gamma F]
                    [--hidden-dim N] [--no-baseline] [--save-path PATH]
                    [--render-every N] [--seed N]

Example
-------
    python train.py --episodes 500 --grid-size 10 --lr 0.001
"""

import argparse
import os
import random

import torch

# Local project imports
from env.simple_world import SimpleWorld
from agents.rl_agent import RLAgent
from utils.logger import Logger


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a REINFORCE agent on SimpleWorld."
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Total number of training episodes (default: 500).",
    )
    parser.add_argument(
        "--grid-size", type=int, default=10,
        help="Number of cells in the 1-D grid world (default: 10).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum steps per episode before truncation (default: 100).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor γ (default: 0.99).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="Hidden layer width of the policy network (default: 64).",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Disable mean-return baseline (higher variance).",
    )
    parser.add_argument(
        "--save-path", type=str, default="policy.pt",
        help="File path to save the final policy weights (default: policy.pt).",
    )
    parser.add_argument(
        "--render-every", type=int, default=0,
        help="Print an ASCII render every N episodes (0 = disabled).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)


def run_episode(env: SimpleWorld, agent: RLAgent, render: bool = False) -> tuple:
    """
    Execute one full episode: collect transitions, store rewards, then update.

    Parameters
    ----------
    env : SimpleWorld
        The environment instance to interact with.
    agent : RLAgent
        The policy-gradient agent.
    render : bool
        If True, print the grid state after every step.

    Returns
    -------
    total_reward : float
        Sum of undiscounted rewards for the episode.
    loss : float
        Policy-gradient loss from the end-of-episode update.
    steps : int
        Number of steps taken before the episode ended.
    goal_reached : bool
        Whether the agent reached the goal cell.
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    goal_reached = False

    while not done:
        # Agent selects action and stores log-probability internally
        action = agent.select_action(obs)

        # Environment transitions to next state
        obs, reward, done, info = env.step(action)

        # Agent stores the received reward
        agent.store_reward(reward)

        total_reward += reward
        steps += 1
        goal_reached = info["goal_reached"]

        if render:
            print(f"  step {steps:>3}: action={'R' if action == 1 else 'L'}  "
                  f"{env.render()}  reward={reward:+.2f}")

    # Perform the policy-gradient update using the stored episode data
    loss = agent.update()

    return total_reward, loss, steps, goal_reached


def main() -> None:
    """Entry point: parse arguments, initialise objects, and run training."""
    args = parse_args()
    set_seed(args.seed)

    # ---- Initialise environment and agent --------------------------------
    env = SimpleWorld(grid_size=args.grid_size, max_steps=args.max_steps)
    agent = RLAgent(
        obs_dim=SimpleWorld.OBS_DIM,
        n_actions=SimpleWorld.N_ACTIONS,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        use_baseline=not args.no_baseline,
    )
    logger = Logger(print_every=50, window=50)

    print("=" * 60)
    print("  Alpha48Alpha AI Lab — REINFORCE Training")
    print("=" * 60)
    print(f"  Grid size   : {args.grid_size}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Episodes    : {args.episodes}")
    print(f"  LR          : {args.lr}")
    print(f"  Gamma       : {args.gamma}")
    print(f"  Hidden dim  : {args.hidden_dim}")
    print(f"  Baseline    : {not args.no_baseline}")
    print(f"  Seed        : {args.seed}")
    print("=" * 60)

    # ---- Training loop ---------------------------------------------------
    for episode in range(1, args.episodes + 1):
        # Optionally render the first step of select episodes
        should_render = args.render_every > 0 and episode % args.render_every == 0

        if should_render:
            print(f"\n--- Rendering episode {episode} ---")

        total_reward, loss, steps, goal_reached = run_episode(
            env, agent, render=should_render
        )

        # Log metrics; summary is printed every 50 episodes
        logger.log_episode(
            episode=episode,
            total_reward=total_reward,
            loss=loss,
            length=steps,
            extra={"goal": goal_reached},
        )

    # ---- End-of-training summary -----------------------------------------
    stats = logger.summary()
    print("\n" + "=" * 60)
    print("  Training complete")
    print(f"  Total episodes    : {stats['total_episodes']}")
    print(f"  Best episode reward: {stats['best_reward']:+.3f}")
    print(f"  Final avg reward  : {stats['final_avg_reward']:+.3f}")
    print(f"  Final avg loss    : {stats['final_avg_loss']:.4f}")
    print("=" * 60)

    # ---- Save final policy weights ---------------------------------------
    agent.save(args.save_path)
    print(f"\n  Policy weights saved to: {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    main()
