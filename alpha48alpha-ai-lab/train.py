"""
train.py — Alpha48Alpha AI Lab
================================
Main training script for the REINFORCE policy-gradient agent on SimpleWorld.

The script trains the agent on a 2-D grid world with obstacles and
optionally co-trains a neural world model on every real transition.

Run
---
    python train.py [--episodes N] [--rows N] [--cols N]
                    [--obstacle-density F] [--max-steps N]
                    [--lr F] [--gamma F] [--hidden-dim N]
                    [--no-baseline] [--train-world-model]
                    [--save-path PATH] [--render-every N] [--seed N]

Example
-------
    python train.py --episodes 800 --rows 6 --cols 6 --obstacle-density 0.15
"""

import argparse
import os
import random

import torch

# Local project imports
from env.simple_world import SimpleWorld, ACTION_LABELS
from agents.rl_agent import RLAgent
from models.world_model import WorldModel
from utils.logger import Logger


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a REINFORCE agent on a 2-D SimpleWorld."
    )
    parser.add_argument(
        "--episodes", type=int, default=800,
        help="Total number of training episodes (default: 800).",
    )
    parser.add_argument(
        "--rows", type=int, default=6,
        help="Number of rows in the 2-D grid world (default: 6).",
    )
    parser.add_argument(
        "--cols", type=int, default=6,
        help="Number of columns in the 2-D grid world (default: 6).",
    )
    parser.add_argument(
        "--obstacle-density", type=float, default=0.15,
        help="Fraction of cells to fill with obstacles (default: 0.15).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Maximum steps per episode before truncation (default: 200).",
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
        help="Hidden layer width for policy and world-model networks (default: 64).",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Disable mean-return baseline (higher variance).",
    )
    parser.add_argument(
        "--train-world-model", action="store_true",
        help="Co-train a neural world model on every real transition.",
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


def run_episode(
    env: SimpleWorld,
    agent: RLAgent,
    world_model: WorldModel | None = None,
    render: bool = False,
) -> tuple:
    """
    Execute one full episode: collect transitions, store rewards, then update.

    If *world_model* is provided, it is updated on every real transition so
    it learns the environment dynamics in parallel with the policy.

    Parameters
    ----------
    env : SimpleWorld
        The 2-D grid world instance to interact with.
    agent : RLAgent
        The policy-gradient agent.
    world_model : WorldModel or None
        Optional neural world model trained on real transitions.
    render : bool
        If True, print the ASCII grid after every step.

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
    wm_loss : float or None
        Average world-model loss over the episode, or None if not trained.
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    goal_reached = False
    wm_losses: list[float] = []

    while not done:
        # Agent selects action and stores log-probability internally
        action = agent.select_action(obs)

        # Environment transitions to next state
        next_obs, reward, done, info = env.step(action)

        # Optionally update the world model on this real transition
        if world_model is not None:
            wm_loss = world_model.train_step(obs, action, next_obs, reward)
            wm_losses.append(wm_loss)

        # Agent stores the received reward
        agent.store_reward(reward)

        if render:
            label = ACTION_LABELS.get(action, str(action))
            print(f"  step {steps + 1:>3}: action={label}  "
                  f"reward={reward:+.2f}  pos={info['position']}")
            print(env.render())

        obs = next_obs
        total_reward += reward
        steps += 1
        goal_reached = info["goal_reached"]

    # Perform the policy-gradient update using the stored episode data
    loss = agent.update()

    avg_wm_loss = (sum(wm_losses) / len(wm_losses)) if wm_losses else None
    return total_reward, loss, steps, goal_reached, avg_wm_loss


def main() -> None:
    """Entry point: parse arguments, initialise objects, and run training."""
    args = parse_args()
    set_seed(args.seed)

    # ---- Initialise environment, agent (and optional world model) --------
    env = SimpleWorld(
        rows=args.rows,
        cols=args.cols,
        max_steps=args.max_steps,
        obstacle_density=args.obstacle_density,
        seed=args.seed,
    )
    agent = RLAgent(
        obs_dim=SimpleWorld.OBS_DIM,
        n_actions=SimpleWorld.N_ACTIONS,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        use_baseline=not args.no_baseline,
    )
    world_model: WorldModel | None = None
    if args.train_world_model:
        world_model = WorldModel(
            obs_dim=SimpleWorld.OBS_DIM,
            n_actions=SimpleWorld.N_ACTIONS,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
        )

    logger = Logger(print_every=50, window=50)

    print("=" * 64)
    print("  Alpha48Alpha AI Lab — REINFORCE Training (2-D Grid)")
    print("=" * 64)
    print(f"  Grid            : {args.rows} × {args.cols}")
    print(f"  Obstacle density: {args.obstacle_density:.0%}")
    print(f"  Max steps       : {args.max_steps}")
    print(f"  Episodes        : {args.episodes}")
    print(f"  LR              : {args.lr}")
    print(f"  Gamma           : {args.gamma}")
    print(f"  Hidden dim      : {args.hidden_dim}")
    print(f"  Baseline        : {not args.no_baseline}")
    print(f"  World model     : {args.train_world_model}")
    print(f"  Seed            : {args.seed}")
    print("=" * 64)

    # Show the initial grid layout so the user can see obstacle placement
    print("\nInitial grid layout:")
    print(env.render())
    print()

    # ---- Training loop ---------------------------------------------------
    for episode in range(1, args.episodes + 1):
        should_render = args.render_every > 0 and episode % args.render_every == 0

        if should_render:
            print(f"\n--- Rendering episode {episode} ---")

        total_reward, loss, steps, goal_reached, avg_wm_loss = run_episode(
            env, agent, world_model=world_model, render=should_render
        )

        # Build optional extra fields for the logger
        extra: dict = {"goal": goal_reached}
        if avg_wm_loss is not None:
            extra["wm_loss"] = f"{avg_wm_loss:.4f}"

        logger.log_episode(
            episode=episode,
            total_reward=total_reward,
            loss=loss,
            length=steps,
            extra=extra,
        )

    # ---- End-of-training summary -----------------------------------------
    stats = logger.summary()
    print("\n" + "=" * 64)
    print("  Training complete")
    print(f"  Total episodes     : {stats['total_episodes']}")
    print(f"  Best episode reward: {stats['best_reward']:+.3f}")
    print(f"  Final avg reward   : {stats['final_avg_reward']:+.3f}")
    print(f"  Final avg loss     : {stats['final_avg_loss']:.4f}")
    print("=" * 64)

    # ---- Save final policy weights ---------------------------------------
    agent.save(args.save_path)
    print(f"\n  Policy weights saved to: {os.path.abspath(args.save_path)}")

    # ---- Optionally save world model weights -----------------------------
    if world_model is not None:
        wm_path = args.save_path.replace(".pt", "_world_model.pt")
        world_model.save(wm_path)
        print(f"  World model saved to   : {os.path.abspath(wm_path)}")


if __name__ == "__main__":
    main()
