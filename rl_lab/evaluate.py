"""Evaluation module.

Loads a trained agent from a checkpoint and runs a deterministic
evaluation, printing per-episode statistics and an optional ASCII
environment render.

CLI usage
---------
::

    python -m rl_lab.evaluate \\
        --checkpoint runs/dqn_gridworld/checkpoint_best.pt \\
        --episodes 20 \\
        --render

Extension hooks
---------------
- Add a ``--record`` flag to write episode videos (requires imageio).
- Override ``_build_env_agent()`` to support custom envs / agents.
- Attach a human observer by pausing between steps and awaiting input.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from rl_lab.agents.dqn import DQNAgent
from rl_lab.agents.reinforce import REINFORCEAgent
from rl_lab.envs.grid_world import GridWorld
from rl_lab.envs.continuous_world import ContinuousWorld
from rl_lab.utils.checkpointing import load_checkpoint


def evaluate(
    checkpoint_path: str,
    n_episodes: int = 20,
    render: bool = False,
) -> dict:
    """Evaluate a trained agent loaded from *checkpoint_path*.

    Parameters
    ----------
    checkpoint_path : str
    n_episodes : int
    render : bool
        Print ASCII env render after each step.

    Returns
    -------
    dict with keys ``mean_return``, ``std_return``, ``success_rate``,
               ``episode_returns``, ``episode_lengths``
    """
    ckpt = load_checkpoint(checkpoint_path)
    config = ckpt["config"]

    # Build env
    env_name = config["env"]
    env_kwargs = config.get("env_kwargs", {})
    if env_name == "gridworld":
        env = GridWorld(**env_kwargs)
    elif env_name == "continuous":
        env = ContinuousWorld(**env_kwargs)
    else:
        raise ValueError(f"Unknown env '{env_name}'")

    # Build agent
    agent_name = config["agent"]
    agent_kwargs = config.get("agent_kwargs", {})
    if agent_name == "dqn":
        agent = DQNAgent(obs_dim=env.obs_dim, act_dim=env.act_dim, **agent_kwargs)
    elif agent_name == "reinforce":
        agent = REINFORCEAgent(
            obs_dim=env.obs_dim, act_dim=env.act_dim,
            is_discrete=env.is_discrete, **agent_kwargs,
        )
    else:
        raise ValueError(f"Unknown agent '{agent_name}'")

    agent.load_state_dict(ckpt["agent_state"])
    agent.set_eval()

    # DQN: force greedy
    if hasattr(agent, "eps_end"):
        agent.eps_end = 0.0  # type: ignore[attr-defined]

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        terminated_ep = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_length += 1
            if terminated:
                terminated_ep = True
            if render:
                print(env.render())
                print()

        returns.append(ep_return)
        lengths.append(ep_length)
        successes.append(terminated_ep)

        print(f"Ep {ep:3d}  return={ep_return:+.3f}  "
              f"length={ep_length}  success={'✓' if terminated_ep else '✗'}")

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    success_rate = float(np.mean(successes))

    print(f"\n{'─'*50}")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Mean return   : {mean_return:+.3f} ± {std_return:.3f}")
    print(f"  Success rate  : {success_rate:.1%}")
    print(f"{'─'*50}\n")

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "success_rate": success_rate,
        "episode_returns": returns,
        "episode_lengths": lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL Lab agent")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true",
                        help="Print ASCII env render during evaluation")
    args = parser.parse_args()
    evaluate(args.checkpoint, n_episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
