"""
config.py — Global hyperparameters and settings for Alpha48Alpha AI Lab.

All tunable values live here so that experiments can be reproduced by
simply archiving this file alongside training logs.
"""

# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------

# Grid dimensions (grid_world.py)
GRID_SIZE = 8          # NxN grid

# World-simulator dimensions (world_simulator.py)
WORLD_WIDTH  = 10
WORLD_HEIGHT = 10

# Number of discrete actions (up / down / left / right)
NUM_ACTIONS = 4

# Reward values
REWARD_GOAL    =  10.0   # agent reached the goal
REWARD_STEP    =  -0.1   # small penalty per time-step to encourage efficiency
REWARD_OBSTACLE = -1.0   # collision with wall or obstacle

# Maximum steps per episode before forced termination
MAX_STEPS_PER_EPISODE = 200

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

NUM_EPISODES      = 300   # total training episodes
LOG_INTERVAL      = 10    # print metrics every N episodes
SAVE_INTERVAL     = 50    # checkpoint model every N episodes
CHECKPOINT_DIR    = "checkpoints"
PLOT_DIR          = "plots"

# ---------------------------------------------------------------------------
# Policy-gradient (REINFORCE) hyperparameters
# ---------------------------------------------------------------------------

PG_LEARNING_RATE = 1e-3
PG_GAMMA         = 0.99   # discount factor
PG_HIDDEN_SIZE   = 128    # hidden layer width for NeuralPolicy

# ---------------------------------------------------------------------------
# DQN hyperparameters
# ---------------------------------------------------------------------------

DQN_LEARNING_RATE    = 5e-4
DQN_GAMMA            = 0.99
DQN_EPSILON_START    = 1.0    # initial exploration rate
DQN_EPSILON_END      = 0.05   # minimum exploration rate
DQN_EPSILON_DECAY    = 0.995  # multiplicative decay per episode
DQN_BATCH_SIZE       = 64
DQN_REPLAY_CAPACITY  = 10_000
DQN_TARGET_UPDATE    = 10     # sync target network every N episodes
DQN_HIDDEN_SIZE      = 128

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
