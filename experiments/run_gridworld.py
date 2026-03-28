#!/usr/bin/env python3
"""
Grid-World RL Experiment
========================
Trains a Q-learning agent on the built-in GridWorld environment and prints
per-episode rewards.  Also builds a TransitionModel alongside the agent so the
learned world model can be inspected after training.

Usage
-----
    python experiments/run_gridworld.py
"""

from ai_lab.environments.grid_world import GridWorld
from ai_lab.agents.q_agent import QAgent
from ai_lab.agents.random_agent import RandomAgent
from ai_lab.world_model.transition_model import TransitionModel

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

GRID_SIZE = 5
N_EPISODES = 500
MAX_STEPS = 200
LOG_EVERY = 50  # print average reward every N episodes

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

env = GridWorld(size=GRID_SIZE)
agent = QAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    learning_rate=0.1,
    discount=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
)
world_model = TransitionModel(n_states=env.n_states, n_actions=env.n_actions)

print(f"Alpha48Alpha AI Lab — Grid-World Experiment")
print(f"Grid size : {GRID_SIZE}×{GRID_SIZE}  ({env.n_states} states, {env.n_actions} actions)")
print(f"Episodes  : {N_EPISODES}  |  Max steps/episode: {MAX_STEPS}")
print("-" * 55)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

episode_rewards = []

for episode in range(1, N_EPISODES + 1):
    state = env.reset()
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.update(state, action, reward, next_state, done)
        world_model.observe(state, action, next_state, reward)

        total_reward += reward
        state = next_state

        if done:
            break

    episode_rewards.append(total_reward)

    if episode % LOG_EVERY == 0:
        recent = episode_rewards[-LOG_EVERY:]
        avg = sum(recent) / len(recent)
        print(
            f"Episode {episode:>4}  |  avg reward (last {LOG_EVERY}): {avg:+.3f}"
            f"  |  ε = {agent.epsilon:.3f}"
        )

# ---------------------------------------------------------------------------
# Final evaluation (greedy policy, no exploration)
# ---------------------------------------------------------------------------

agent.epsilon = 0.0
eval_rewards = []
EVAL_EPISODES = 20

for _ in range(EVAL_EPISODES):
    state = env.reset()
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    eval_rewards.append(total_reward)

avg_eval = sum(eval_rewards) / len(eval_rewards)
solved = avg_eval > 0.5

print("-" * 55)
print(f"Evaluation over {EVAL_EPISODES} greedy episodes: avg reward = {avg_eval:+.3f}")
print(
    f"World model — transitions observed: "
    f"{sum(world_model.transition_count(s, a) for s in range(env.n_states) for a in range(env.n_actions))}"
)
print(f"Result: {'✅  Solved!' if solved else '🔄  Still learning — try more episodes.'}")
