# Alpha48Alpha AI Lab

> **Experimental AI Research** — Reinforcement Learning · World Models · Simulation Environments

---

## Overview

Alpha48Alpha AI Lab is a modular Python research framework for experimenting with reinforcement learning algorithms and simulation environments. It is designed as a starting point for world-model research, where an agent must learn to understand and predict its environment in order to act effectively.

---

## Features

| Module | Description |
|--------|-------------|
| `env/simple_world.py` | 2D grid world environment with obstacles, goals, and a standard RL interface |
| `agents/rl_agent.py` | REINFORCE policy-gradient agent implemented in PyTorch |
| `utils/logger.py` | Lightweight training logger with CSV export and console output |
| `train.py` | End-to-end training script with configurable hyperparameters |

---

## Project Structure

```
alpha48alpha-ai-lab/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── train.py            # Main training script
├── env/
│   ├── __init__.py
│   └── simple_world.py # 2D grid world simulation environment
├── agents/
│   ├── __init__.py
│   └── rl_agent.py     # REINFORCE policy gradient agent (PyTorch)
└── utils/
    ├── __init__.py
    └── logger.py       # Training metrics logger
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- [PyTorch](https://pytorch.org/get-started/locally/) (CPU build is sufficient for these experiments)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Alpha48Alpha/Ai-montizable-.git
cd Ai-montizable-/alpha48alpha-ai-lab

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the agent

```bash
python train.py
```

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 500 | Number of training episodes |
| `--grid-size` | 5 | Side length of the grid (5 → 5×5 grid) |
| `--max-steps` | 200 | Maximum steps per episode |
| `--lr` | 0.001 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--render-every` | 0 | Print the grid every N episodes (0 = off) |
| `--save-path` | `policy.pt` | Path to save the best policy weights |

### Example run

```bash
python train.py --episodes 1000 --grid-size 7 --render-every 200 --lr 5e-4
```

---

## How It Works

### Environment (`SimpleWorld`)

The agent lives on a square grid. Each cell is either:

- **Empty** (`.`) — the agent can move here freely.
- **Obstacle** (`#`) — blocked; bumping into one gives a small negative reward.
- **Goal** (`G`) — reaching here ends the episode with a positive reward.

The agent observes its current position as a one-hot vector and can take four actions: **up**, **down**, **left**, **right**.

### Agent (`PolicyGradientAgent`)

A two-layer fully-connected neural network maps the observation to a probability distribution over actions. After each episode the **REINFORCE** update rule adjusts the network weights:

```
∇J(θ) = E[ ∇ log π(a|s; θ) · G_t ]
```

where `G_t` is the discounted cumulative return from timestep `t`.

### Logger (`Logger`)

Each episode's `total_reward`, `loss`, `episode_length`, and elapsed time are printed to the console and appended to `logs/pg_training.csv` for later analysis.

---

## Extending the Lab

This project is intentionally minimal. Suggested next steps:

- **Add a world model** — train a neural network to predict the next observation given the current observation and action.
- **Implement actor-critic** — add a value-function baseline to reduce gradient variance.
- **Larger / procedural environments** — swap `SimpleWorld` for a more complex environment.
- **Experiment tracking** — integrate [Weights & Biases](https://wandb.ai/) or [TensorBoard](https://www.tensorflow.org/tensorboard).

---

## License

This project is released under the [MIT License](../LICENSE).
