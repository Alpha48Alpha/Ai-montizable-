"""
config.py — Central configuration for Alpha48Alpha AI Lab.

All hyperparameters and environment settings are defined here so that
every module can import a single source of truth.
"""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
GRID_SIZE: int = 8          # Side length of the 2-D grid world (grid is GRID_SIZE × GRID_SIZE)
MAX_STEPS: int = 200        # Maximum steps per episode before forced termination

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_EPISODES: int = 1_000   # Total training episodes
LEARNING_RATE: float = 1e-3 # Adam optimiser learning rate
GAMMA: float = 0.99         # Discount factor for future rewards
BATCH_SIZE: int = 64        # Mini-batch size used in DQN replay training
REPLAY_BUFFER_SIZE: int = 10_000  # Maximum transitions stored in the replay buffer
TARGET_UPDATE_FREQ: int = 20      # How often (in episodes) to sync the DQN target network
EPSILON_START: float = 1.0        # Initial exploration probability (ε-greedy)
EPSILON_END: float = 0.05         # Minimum exploration probability
EPSILON_DECAY: float = 0.995      # Per-episode multiplicative decay of ε

# ---------------------------------------------------------------------------
# Logging / visualization
# ---------------------------------------------------------------------------
LOG_INTERVAL: int = 50      # Print training stats every N episodes
RENDER_INTERVAL: int = 100  # Render grid world every N episodes (0 = never)
