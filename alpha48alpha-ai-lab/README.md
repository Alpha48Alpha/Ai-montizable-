# Alpha48Alpha AI Lab

A modular **reinforcement learning research project** built with PyTorch.
The project provides a clean foundation for policy-gradient experiments and is
designed to be extended with world models, model-based RL, and more advanced
agent architectures.

---

## Project Structure

```
alpha48alpha-ai-lab/
├── README.md               ← This file
├── requirements.txt        ← Python dependencies
├── train.py                ← Main training script
├── env/
│   ├── __init__.py
│   └── simple_world.py     ← 2-D grid world with obstacles
├── agents/
│   ├── __init__.py
│   └── rl_agent.py         ← REINFORCE policy-gradient agent (PyTorch)
├── models/
│   ├── __init__.py
│   └── world_model.py      ← Neural forward model (obs, action) → (next_obs, reward)
└── utils/
    ├── __init__.py
    └── logger.py           ← Modular training logger
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

implemented as minimising `-J(θ)` via Adam.

---

## Environment — SimpleWorld (2-D)

`SimpleWorld` is a 2-D grid with configurable size and randomly placed obstacles.

| Property           | Value                                               |
|--------------------|-----------------------------------------------------|
| Observation        | Normalised `[row, col]` of the agent — shape `[2]` |
| Actions            | 0 = up, 1 = down, 2 = left, 3 = right              |
| Start              | Top-left cell `(0, 0)`                              |
| Goal               | Bottom-right cell `(rows-1, cols-1)`                |
| Reward — goal      | +1.0                                                |
| Reward — obstacle  | −0.5 (agent stays put)                              |
| Reward — step      | −0.01 (time penalty)                                |
| Terminal           | Goal reached **or** `max_steps` exceeded            |

### ASCII render legend

```
+------+
|A.....|   A = agent
|..#...|   # = obstacle
|...#..|   G = goal
|....G.|   . = empty cell
+------+   * = agent on goal
```

---

## World Model

`models/world_model.py` implements a neural **forward model** that learns:

```
(obs_t, action_t)  →  (obs_{t+1},  reward_t)
```

Enable it during training with `--train-world-model`.  Once trained it can
generate *imagined roll-outs* without touching the real environment — the
core idea behind Dreamer, RSSM, and MuZero.

**Architecture:**

```
[obs ‖ action_one_hot] → Linear(hidden) → ReLU → Linear(hidden) → ReLU
                         ↓                                         ↓
                     obs_delta head                          reward head
```

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0 is required (CPU-only install is sufficient).

---

## Quick Start

```bash
# Default run — 6×6 grid, 800 episodes
python train.py

# Larger grid with obstacle rendering every 200 episodes
python train.py --rows 8 --cols 8 --obstacle-density 0.2 --render-every 200

# Co-train a world model alongside the policy
python train.py --train-world-model

# Disable variance-reduction baseline (pure REINFORCE)
python train.py --no-baseline --lr 0.005
```

### All CLI flags

| Flag                  | Default    | Description                                          |
|-----------------------|------------|------------------------------------------------------|
| `--episodes`          | 800        | Total training episodes                              |
| `--rows`              | 6          | Grid rows                                            |
| `--cols`              | 6          | Grid columns                                         |
| `--obstacle-density`  | 0.15       | Fraction of cells filled with obstacles (0.0–0.8)   |
| `--max-steps`         | 200        | Steps before episode truncation                      |
| `--lr`                | 0.001      | Adam learning rate                                   |
| `--gamma`             | 0.99       | Discount factor γ                                    |
| `--hidden-dim`        | 64         | Hidden layer width (policy + world model)            |
| `--no-baseline`       | off        | Disable mean-return baseline                         |
| `--train-world-model` | off        | Co-train a neural world model on real transitions    |
| `--save-path`         | policy.pt  | Path to save final policy weights                    |
| `--render-every`      | 0          | Print ASCII grid render every N episodes             |
| `--seed`              | 42         | Random seed for reproducibility                      |

---

## Expected Output

```
================================================================
  Alpha48Alpha AI Lab — REINFORCE Training (2-D Grid)
================================================================
  Grid            : 6 × 6
  Obstacle density: 15%
  ...
================================================================

Initial grid layout:
+------+
|A.....|
|..#...|
|...#..|
|....#.|
|......|
|.....G|
+------+

[Ep    50]  avg_reward=-1.990  avg_loss=0.0821  avg_steps=200.0  goal=False
[Ep   100]  avg_reward=-0.440  avg_loss=0.0312  avg_steps=45.0   goal=True
...
[Ep   800]  avg_reward=+0.870  avg_loss=0.0038  avg_steps=13.0   goal=True

================================================================
  Training complete
  Total episodes     : 800
  Best episode reward: +1.000
  Final avg reward   : +0.870
  Final avg loss     : 0.0038
================================================================
  Policy weights saved to: /path/to/alpha48alpha-ai-lab/policy.pt
```

---

## Extending the Project

| Goal                          | Where to start                                          |
|-------------------------------|---------------------------------------------------------|
| New environment               | Add a class in `env/` following `SimpleWorld`'s interface |
| Actor-Critic (A2C)            | Extend `RLAgent` with a `ValueNetwork`                  |
| PPO / TRPO                    | Replace the loss function in `RLAgent.update()`         |
| Recurrent world model (RSSM)  | Extend `WorldModel` with a GRU encoder                  |
| DreamerV3-style training      | Use `WorldModel.predict()` to generate imagined episodes |
| Logging to CSV / TensorBoard  | Extend `Logger` in `utils/logger.py`                    |

---

## References

- Williams, R.J. (1992). *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning.* Machine Learning, 8, 229–256.
- Ha, D. & Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122.
- Hafner, D. et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3).* arXiv:2301.04104.
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction.*
  MIT Press. [http://incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html)
