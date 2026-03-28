# Alpha48Alpha AI Lab

> Exploring reinforcement learning, world models, and creative AI systems.

---

## About

**Alpha48Alpha AI Lab** is an open research playground for building and experimenting
with AI systems that learn through simulation and interaction. The lab combines
classical reinforcement learning, modern world-modeling techniques, and creative
generative AI into a single, accessible codebase.

---

## Research Areas

| Area | Description |
|------|-------------|
| 🤖 **Reinforcement Learning** | Agents that learn optimal behaviour by interacting with environments |
| 🌍 **World Models** | Learned internal representations that let agents plan without acting in the real world |
| 🎨 **Generative Models** | Producing novel content — images, video, text — through learned distributions |
| 🧭 **AI Alignment** | Techniques for keeping agent objectives consistent with human intent |
| 🏗 **Simulation Environments** | Custom environments for training and evaluating AI agents |

---

## Goal

Build experimental AI systems that **learn through simulation and interaction** —
systems that can model their environment, plan ahead, and produce creative outputs.

---

## Repository Structure

```
Ai-montizable-/
├── README.md                    # This file — lab overview
│
├── ai_lab/                      # Core Python library
│   ├── environments/            # Simulation environments
│   │   ├── base_env.py          # Abstract base environment (gym-style)
│   │   └── grid_world.py        # Simple grid-world for RL experiments
│   ├── agents/                  # RL agents
│   │   ├── base_agent.py        # Abstract base agent
│   │   ├── random_agent.py      # Random baseline agent
│   │   └── q_agent.py           # Tabular Q-learning agent
│   └── world_model/             # World-model components
│       ├── base_model.py        # Abstract world-model interface
│       └── transition_model.py  # Learned transition / next-state predictor
│
├── experiments/
│   └── run_gridworld.py         # End-to-end RL experiment on the grid world
│
└── templates/                   # Creative AI production templates
    ├── movie_production_package.md
    ├── character_sheet.md
    └── scene_breakdown.md
```

---

## Quick Start

```bash
# Run the grid-world reinforcement-learning experiment
python experiments/run_gridworld.py
```

The script trains a Q-learning agent on the built-in `GridWorld` environment,
prints per-episode rewards, and reports final performance.

---

## Core Concepts

### Environments

All environments inherit from `ai_lab.environments.BaseEnv` and expose a
gym-style interface:

```python
from ai_lab.environments.grid_world import GridWorld

env = GridWorld(size=5)
obs = env.reset()
obs, reward, done, info = env.step(action=0)
```

### Agents

All agents inherit from `ai_lab.agents.BaseAgent`:

```python
from ai_lab.agents.q_agent import QAgent

agent = QAgent(n_states=25, n_actions=4, learning_rate=0.1, discount=0.99)
action = agent.select_action(state=obs)
agent.update(state=obs, action=action, reward=reward, next_state=next_obs, done=done)
```

### World Models

World models learn to predict the next state and reward given a current state
and action, enabling model-based planning without extra environment steps:

```python
from ai_lab.world_model.transition_model import TransitionModel

model = TransitionModel(n_states=25, n_actions=4)
model.observe(state, action, next_state, reward)
predicted_next, predicted_reward = model.predict(state, action)
```

---

## Creative AI Systems

The repository also includes a full **animation production engine** that applies
generative AI to creative content — a practical example of generative models in
action.  See [`movie_engine_config.md`](movie_engine_config.md) and
[`templates/`](templates/) for details, or generate a sample production package:

```bash
python movie_engine.py
```

---

## Contributing

Contributions are welcome. Open an issue or pull request to propose new
environments, agent algorithms, world-model architectures, or alignment
experiments.

---

## License

MIT

