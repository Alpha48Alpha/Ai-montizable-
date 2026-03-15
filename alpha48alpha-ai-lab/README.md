# Alpha48Alpha AI Lab

A modular **reinforcement learning research project** built with PyTorch.
The project provides a clean foundation for policy-gradient experiments and is
designed to be extended with world models, model-based RL, and more advanced
agent architectures.

---

## Project Structure

```
alpha48alpha-ai-lab/
├── README.md           ← This file
├── requirements.txt    ← Python dependencies
├── train.py            ← Main training script
├── env/
│   ├── __init__.py
│   └── simple_world.py ← 1-D grid world environment
├── agents/
│   ├── __init__.py
│   └── rl_agent.py     ← REINFORCE policy-gradient agent (PyTorch)
└── utils/
    ├── __init__.py
    └── logger.py       ← Modular training logger
```

---

## Algorithm

The agent uses **REINFORCE** (vanilla policy gradient, Williams 1992):

1. Collect a full episode of `(state, action, reward)` transitions using the current policy `π_θ`.
2. Compute discounted returns `G_t = Σ_{k=0}^{T-t} γ^k · r_{t+k}` backwards through the trajectory.
3. Optionally subtract the mean return as a **baseline** to reduce gradient variance.
4. Update the policy by ascending the gradient of the objective:

```
J(θ) = E[ Σ_t  log π_θ(a_t | s_t) · G_t ]
```

which is implemented as minimising `-J(θ)` via Adam.

---

## Environment

`SimpleWorld` is a 1-D grid with `grid_size` cells.

| Property    | Value                                     |
|-------------|-------------------------------------------|
| Observation | Normalised agent position `[0.0 … 1.0]`  |
| Actions     | 0 = move left, 1 = move right             |
| Reward      | +1.0 at goal cell, −0.01 per step         |
| Terminal    | Goal reached **or** `max_steps` exceeded  |

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0 is required (CPU-only install is sufficient for this project).

---

## Quick Start

```bash
# Train for 500 episodes with default settings
python train.py

# Custom run — larger grid, more episodes, render every 100 episodes
python train.py --episodes 1000 --grid-size 15 --render-every 100

# Disable variance-reduction baseline (pure REINFORCE)
python train.py --no-baseline --lr 0.005
```

### All CLI flags

| Flag            | Default    | Description                                  |
|-----------------|------------|----------------------------------------------|
| `--episodes`    | 500        | Total training episodes                      |
| `--grid-size`   | 10         | Number of grid cells                         |
| `--max-steps`   | 100        | Steps before episode truncation              |
| `--lr`          | 0.001      | Adam learning rate                           |
| `--gamma`       | 0.99       | Discount factor γ                            |
| `--hidden-dim`  | 64         | Policy network hidden layer width            |
| `--no-baseline` | off        | Disable mean-return baseline                 |
| `--save-path`   | policy.pt  | Path to save final policy weights            |
| `--render-every`| 0          | Print ASCII grid render every N episodes     |
| `--seed`        | 42         | Random seed for reproducibility              |

---

## Expected Output

```
============================================================
  Alpha48Alpha AI Lab — REINFORCE Training
============================================================
  Grid size   : 10
  Max steps   : 100
  Episodes    : 500
  LR          : 0.001
  Gamma       : 0.99
  Hidden dim  : 64
  Baseline    : True
  Seed        : 42
============================================================
[Ep    50]  avg_reward=-0.490  avg_loss=0.0312  avg_steps=49.0  goal=False
[Ep   100]  avg_reward=+0.120  avg_loss=0.0185  avg_steps=18.0  goal=True
...
[Ep   500]  avg_reward=+0.910  avg_loss=0.0041  avg_steps=9.0   goal=True

============================================================
  Training complete
  Total episodes    : 500
  Best episode reward: +1.000
  Final avg reward  : +0.910
  Final avg loss    : 0.0041
============================================================
  Policy weights saved to: /path/to/alpha48alpha-ai-lab/policy.pt
```

---

## Extending the Project

| Goal                         | Where to start                                  |
|------------------------------|-------------------------------------------------|
| New environment              | Add a class in `env/` following `SimpleWorld`'s interface |
| Actor-Critic (A2C)           | Extend `RLAgent` with a `ValueNetwork`          |
| PPO / TRPO                   | Replace the loss in `RLAgent.update()`          |
| World model (RSSM / DreamerV3) | Add `models/world_model.py`                   |
| Logging to CSV / TensorBoard | Extend `Logger` in `utils/logger.py`            |

---

## References

- Williams, R.J. (1992). *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning.* Machine Learning, 8, 229–256.
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction.*
  MIT Press. [http://incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html)
