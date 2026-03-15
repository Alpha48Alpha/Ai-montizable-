"""
world_simulator.py — Higher-level world simulator wrapping GridWorld.

WorldSimulator adds:
    - episode bookkeeping (total reward, step count)
    - a convenience ``run_episode`` method for full-episode rollouts
    - a ``collect_trajectory`` method that returns raw (s, a, r, s', done) tuples
      suitable for replay buffers and policy-gradient training.
"""

from typing import Callable, List, Optional, Tuple

from alpha48alpha_ai_lab.environments.grid_world import GridWorld


Transition = Tuple[int, int, float, int, bool]   # (state, action, reward, next_state, done)


class WorldSimulator:
    """Wraps GridWorld and provides higher-level episode utilities."""

    def __init__(
        self,
        grid_size: int = 8,
        num_obstacles: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the world simulator.

        Args:
            grid_size:      Side length of the square grid world.
            num_obstacles:  Number of obstacles to place each episode.
            seed:           Optional RNG seed for reproducibility.
        """
        self.env = GridWorld(grid_size=grid_size, num_obstacles=num_obstacles, seed=seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """Reset the underlying environment and return the initial state."""
        return self.env.reset()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Forward a single action to the environment."""
        return self.env.step(action)

    def render(self) -> None:
        """Render the current environment state."""
        self.env.render()

    def get_state(self) -> int:
        """Return the current observation."""
        return self.env.get_state()

    # ------------------------------------------------------------------
    # Episode utilities
    # ------------------------------------------------------------------

    def run_episode(
        self,
        policy_fn: Callable[[int], int],
        render: bool = False,
    ) -> Tuple[float, int]:
        """Run a full episode using *policy_fn* and return (total_reward, steps).

        Args:
            policy_fn:  Callable that maps a state (int) to an action (int).
            render:     Whether to call ``render()`` at every step.

        Returns:
            Tuple of (total_reward, num_steps).
        """
        state = self.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = policy_fn(state)
            next_state, reward, done, _ = self.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            if render:
                self.render()

        return total_reward, steps

    def collect_trajectory(
        self,
        policy_fn: Callable[[int], int],
    ) -> List[Transition]:
        """Collect a full episode trajectory.

        Returns:
            List of (state, action, reward, next_state, done) tuples.
        """
        state = self.reset()
        trajectory: List[Transition] = []
        done = False

        while not done:
            action = policy_fn(state)
            next_state, reward, done, _ = self.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state

        return trajectory

    # ------------------------------------------------------------------
    # Properties forwarded from inner env
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """Total number of distinct states."""
        return self.env.state_size

    @property
    def num_actions(self) -> int:
        """Number of discrete actions."""
        return GridWorld.NUM_ACTIONS
