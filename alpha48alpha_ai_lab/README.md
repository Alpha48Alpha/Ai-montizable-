# Alpha48Alpha AI Lab

> **Experimental reinforcement learning research environment**

A modular Python lab for training RL agents in simulated environments.
Designed for rapid experimentation with policy-gradient methods, deep
Q-learning, and future world-model architectures.

---

## Features

| Module | Description |
|---|---|
| `environments/` | 2-D grid world + higher-level world simulator |
| `agents/` | REINFORCE (policy gradient) and DQN agents |
| `models/` | Neural policy and Q-network (PyTorch) |
| `training/` | Unified trainer and experience replay buffer |
| `utils/` | Metrics logger and matplotlib visualisation |

---

## Project Structure

```
alpha48alpha_ai_lab/
├── train.py                       # Training entry-point
├── config.py                      # All hyperparameters
├── requirements.txt
│
├── agents/
│   ├── policy_gradient_agent.py   # REINFORCE agent
│   └── dqn_agent.py               # Deep Q-Network agent
│
├── environments/
│   ├── grid_world.py              # 2-D grid world
│   └── world_simulator.py         # Episode-level wrapper
│
├── models/
│   ├── neural_policy.py           # Softmax policy network
│   └── value_network.py           # V(s) and Q(s,a) networks
│
├── training/
│   ├── trainer.py                 # Unified training loop
│   └── replay_buffer.py           # Experience replay buffer
│
└── utils/
    ├── logger.py                  # CSV + console metrics logger
    └── visualization.py           # Grid renderer & reward curve plotter
```

---

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r alpha48alpha_ai_lab/requirements.txt
```

---

## Quick Start

```bash
# Train with DQN (default)
python -m alpha48alpha_ai_lab.train

# Train with policy gradient
python -m alpha48alpha_ai_lab.train --agent pg

# 500 episodes, save metrics and curve
python -m alpha48alpha_ai_lab.train --agent dqn --episodes 500 \
    --csv logs/run.csv --curve logs/curve.png
```

Or run the entry-point script directly from the repo root:

```bash
python alpha48alpha_ai_lab/train.py --agent dqn --episodes 1000
```

---

## Environment

The **GridWorld** environment is a fully observable, discrete 2-D grid:

- Agent starts at `(0, 0)`, goal is at `(N-1, N-1)`.
- Actions: UP / DOWN / LEFT / RIGHT (walls are solid).
- Reaching the goal: **+1** reward, episode ends.
- Hitting an obstacle: **−1** reward, episode ends.
- Each step: **−0.01** living penalty (encourages efficiency).

---

## Agents

### REINFORCE (Policy Gradient)
Classic Monte-Carlo policy gradient with reward normalisation.
The policy is a two-layer feed-forward network with softmax output.

### DQN (Deep Q-Network)
Implements the original DQN paper (Mnih et al., 2015) with:
- Experience replay buffer
- Separate target network (periodically synced)
- ε-greedy exploration with exponential decay

---

## Configuration

All hyperparameters are in `config.py`:

```python
NUM_EPISODES      = 1_000
LEARNING_RATE     = 1e-3
GAMMA             = 0.99
BATCH_SIZE        = 64
REPLAY_BUFFER_SIZE = 10_000
EPSILON_START     = 1.0
EPSILON_END       = 0.05
EPSILON_DECAY     = 0.995
```

---

## Roadmap

- [ ] World model (latent-space dynamics model)
- [ ] Actor-Critic (A2C)
- [ ] Continuous action spaces
- [ ] Multi-agent experiments
- [ ] Curiosity-driven exploration
