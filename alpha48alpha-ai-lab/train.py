"""
Alpha48Alpha AI Lab — Training Script
======================================
Trains a REINFORCE policy-gradient agent in the SimpleWorld environment.

Usage
-----
    python train.py [--episodes N] [--grid-size G] [--max-steps M]
                    [--lr LR] [--gamma GAMMA] [--seed SEED]
                    [--save-path PATH] [--render-every N]

Example
-------
    python train.py --episodes 500 --grid-size 5 --render-every 100
"""

import argparse
import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from agents.rl_agent import PolicyGradientAgent
from env.simple_world import SimpleWorld
from utils.logger import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a policy gradient agent in SimpleWorld."
    )
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes (default: 500)")
    parser.add_argument("--grid-size", type=int, default=5,
                        help="Side length of the grid world (default: 5)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum steps per episode (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the policy network (default: 0.001)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for returns (default: 0.99)")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="Hidden layer width of the policy network (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the environment (default: 42)")
    parser.add_argument("--save-path", type=str, default="policy.pt",
                        help="Where to save the trained policy weights (default: policy.pt)")
    parser.add_argument("--render-every", type=int, default=0,
                        help="Render the environment every N episodes; 0 = never (default: 0)")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Main training loop.

    Creates the environment and agent, runs episodes, and updates the policy
    at the end of each episode using the REINFORCE gradient estimator.
    """
    print("\n🤖  Alpha48Alpha AI Lab — Policy Gradient Training")
    print(f"    Grid: {args.grid_size}×{args.grid_size}  |  "
          f"Episodes: {args.episodes}  |  Seed: {args.seed}")

    # Instantiate environment
    env = SimpleWorld(
        grid_size=args.grid_size,
        seed=args.seed,
    )

    # Instantiate agent
    agent = PolicyGradientAgent(
        state_size=env.state_size,
        action_size=SimpleWorld.NUM_ACTIONS,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
    )

    # Instantiate logger
    logger = Logger(log_dir="logs", experiment_name="pg_training")
    logger.print_header()

    best_reward = float("-inf")

    for episode in range(args.episodes):
        observation = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        # ----- Episode rollout -----
        while not done and steps < args.max_steps:
            # Select action from the current policy
            action = agent.select_action(observation)

            # Take a step in the environment
            observation, reward, done = env.step(action)

            # Record reward for policy gradient update
            agent.store_reward(reward)

            total_reward += reward
            steps += 1

        # ----- Optional rendering -----
        if args.render_every > 0 and (episode + 1) % args.render_every == 0:
            print(f"\n  [Render — episode {episode + 1}]")
            env.render()

        # ----- Policy update -----
        loss = agent.update_policy()

        # ----- Logging -----
        logger.log_episode(
            episode=episode + 1,
            total_reward=total_reward,
            loss=loss,
            episode_length=steps,
        )

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(args.save_path)

    logger.print_summary()
    logger.close()

    print(f"  ✅  Training complete. Best episode reward: {best_reward:.3f}")
    print(f"  💾  Policy saved to '{args.save_path}'")


if __name__ == "__main__":
    train(parse_args())
