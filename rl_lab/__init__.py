"""
RL Lab — Production-grade AI Research Lab
==========================================
Modular PyTorch reinforcement-learning framework featuring:
  - Rich simulated worlds (GridWorld, ContinuousWorld)
  - Policy-gradient (REINFORCE) and DQN agents
  - Experiment configs, checkpointing, and metrics logging
  - Evaluation and visualization helpers
  - Easy extension points for world models, multimodal control,
    and human-in-the-loop research

Quick start
-----------
    python train.py --config rl_lab/configs/dqn_gridworld.json
    python train.py --config rl_lab/configs/reinforce_gridworld.json
    python train.py --config rl_lab/configs/reinforce_continuous.json
"""

__version__ = "1.0.0"
