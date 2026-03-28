"""
train.py — Entry-point for training RL agents in Alpha48Alpha AI Lab.

Usage examples
--------------
# Train the default policy-gradient agent in GridWorld:
    python train.py

# Train the DQN agent in the world simulator:
    python train.py --agent dqn --env world_simulator

# Override the number of episodes:
    python train.py --episodes 500
"""

import argparse
import os
import random

import numpy as np
import torch

import config
from environments.grid_world import GridWorld
from environments.world_simulator import WorldSimulator
from agents.policy_gradient_agent import PolicyGradientAgent
from agents.dqn_agent import DQNAgent
from training.trainer import Trainer
from utils.logger import Logger
from utils.visualization import plot_rewards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_environment(name: str):
    """Instantiate an environment by short name."""
    if name == "grid_world":
        return GridWorld(size=config.GRID_SIZE)
    if name == "world_simulator":
        return WorldSimulator(
            width=config.WORLD_WIDTH,
            height=config.WORLD_HEIGHT,
        )
    raise ValueError(f"Unknown environment: '{name}'. "
                     f"Choose from: grid_world, world_simulator")


def build_agent(name: str, state_size: int, action_size: int):
    """Instantiate an agent by short name."""
    if name == "policy_gradient":
        return PolicyGradientAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=config.PG_LEARNING_RATE,
            gamma=config.PG_GAMMA,
            hidden_size=config.PG_HIDDEN_SIZE,
        )
    if name == "dqn":
        return DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=config.DQN_LEARNING_RATE,
            gamma=config.DQN_GAMMA,
            epsilon=config.DQN_EPSILON_START,
            epsilon_min=config.DQN_EPSILON_END,
            epsilon_decay=config.DQN_EPSILON_DECAY,
            batch_size=config.DQN_BATCH_SIZE,
            replay_capacity=config.DQN_REPLAY_CAPACITY,
            target_update=config.DQN_TARGET_UPDATE,
            hidden_size=config.DQN_HIDDEN_SIZE,
        )
    raise ValueError(f"Unknown agent: '{name}'. "
                     f"Choose from: policy_gradient, dqn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alpha48Alpha AI Lab — train RL agent")
    parser.add_argument("--agent",    default="policy_gradient",
                        choices=["policy_gradient", "dqn"],
                        help="Agent algorithm to use (default: policy_gradient)")
    parser.add_argument("--env",      default="grid_world",
                        choices=["grid_world", "world_simulator"],
                        help="Environment to train in (default: grid_world)")
    parser.add_argument("--episodes", type=int, default=config.NUM_EPISODES,
                        help=f"Number of training episodes (default: {config.NUM_EPISODES})")
    parser.add_argument("--render",   action="store_true",
                        help="Render the environment during training")
    args = parser.parse_args()

    # Reproducibility
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    # Build environment
    env = build_environment(args.env)
    state = env.reset()
    state_size  = len(state)
    action_size = config.NUM_ACTIONS

    print(f"\n{'='*60}")
    print(f"  Alpha48Alpha AI Lab")
    print(f"  Agent       : {args.agent}")
    print(f"  Environment : {args.env}")
    print(f"  State size  : {state_size}")
    print(f"  Action size : {action_size}")
    print(f"  Episodes    : {args.episodes}")
    print(f"{'='*60}\n")

    # Build agent
    agent = build_agent(args.agent, state_size, action_size)

    # Build logger
    logger = Logger(agent_name=args.agent, env_name=args.env)

    # Build trainer and run
    trainer = Trainer(
        agent=agent,
        env=env,
        logger=logger,
        num_episodes=args.episodes,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        log_interval=config.LOG_INTERVAL,
        save_interval=config.SAVE_INTERVAL,
        checkpoint_dir=config.CHECKPOINT_DIR,
        render=args.render,
    )
    episode_rewards = trainer.train()

    # Save reward curve
    plot_path = os.path.join(config.PLOT_DIR, f"{args.agent}_{args.env}_rewards.png")
    plot_rewards(episode_rewards, title=f"{args.agent} on {args.env}",
                 save_path=plot_path)
    print(f"\nReward curve saved to {plot_path}")


if __name__ == "__main__":
    main()
