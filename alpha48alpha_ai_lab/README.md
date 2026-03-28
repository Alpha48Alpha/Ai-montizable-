# Alpha48Alpha AI Lab

A modular **reinforcement learning research laboratory** built with Python and PyTorch.
The lab provides a clean, production-style foundation for experimenting with RL agents
in simulated environments.

---

## Project Structure

```
alpha48alpha_ai_lab/
├── README.md               — this file
├── requirements.txt        — Python dependencies
├── config.py               — hyperparameters & global settings
├── train.py                — entry-point: launch a training run
│
├── agents/
│   ├── policy_gradient_agent.py   — REINFORCE / policy-gradient agent
│   └── dqn_agent.py               — Deep Q-Network (DQN) agent
│
├── environments/
│   ├── grid_world.py              — minimal grid-world environment
│   └── world_simulator.py        — 2-D world with obstacles & goal states
│
├── models/
│   ├── neural_policy.py           — stochastic actor policy network
│   └── value_network.py          — critic / Q-value network
│
├── training/
│   ├── trainer.py                 — episode collection & policy update loop
│   └── replay_buffer.py          — experience replay buffer (for DQN)
│
└── utils/
    ├── logger.py                  — reward & metric logging
    └── visualization.py          — matplotlib agent visualisation
```

---

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the default agent (policy gradient in GridWorld)
python train.py

# 4. Train with a different agent / environment
python train.py --agent dqn --env world_simulator --episodes 500
```

---

## Agents

| Agent | Algorithm | File |
|-------|-----------|------|
| `policy_gradient` | REINFORCE (Monte-Carlo policy gradient) | `agents/policy_gradient_agent.py` |
| `dqn` | Deep Q-Network with experience replay & target network | `agents/dqn_agent.py` |

## Environments

| Environment | Description | File |
|-------------|-------------|------|
| `grid_world` | Simple N×N grid, no obstacles | `environments/grid_world.py` |
| `world_simulator` | 2-D world with walls, obstacles, and multiple goal states | `environments/world_simulator.py` |

---

## Configuration

All hyperparameters live in `config.py`. Edit that file to change learning rates,
network sizes, episode counts, and environment dimensions without touching agent code.

---

## Expected Training Output

```
Episode   10 | Reward:  -23.00 | Steps:  45 | Epsilon: 0.950
Episode   20 | Reward:  -18.00 | Steps:  38 | Epsilon: 0.905
...
Episode  200 | Reward:    5.00 | Steps:  12 | Epsilon: 0.135
Training complete. Model saved to checkpoints/
```

Reward curves and agent trajectories are saved automatically to `plots/`.
